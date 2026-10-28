import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from faster_whisper.tokenizer import _LANGUAGE_CODES

import app


class FakeSegment:
    def __init__(self, text="recognized speech", start=1.0, end=2.0):
        self.text = text
        self.start = start
        self.end = end


class FakeModel:
    def transcribe(self, audio, **kwargs):
        return (
            iter([FakeSegment()]),
            SimpleNamespace(language="te", language_probability=0.98),
        )


class TranscriptionTests(unittest.TestCase):
    def make_wav(self, directory, seconds=2):
        audio_path = Path(directory) / "sample.wav"
        with wave.open(str(audio_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(app.SAMPLE_RATE)
            audio_file.writeframes(b"\x00\x00" * app.SAMPLE_RATE * seconds)
        return audio_path

    def test_language_choices_cover_whisper_codes(self):
        self.assertEqual(set(app.LANGUAGE_NAMES), set(_LANGUAGE_CODES))
        self.assertEqual(len(app.SUPPORTED_LANGUAGE_CHOICES), 100)

    def test_transcription_streams_partial_text_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = self.make_wav(directory)
            with patch.object(app, "get_model", return_value=FakeModel()):
                updates = list(app.transcribe_audio(str(audio_path), model_size="tiny"))

        self.assertIn("Preparing your audio", updates[0][2])
        self.assertIn("Processing chunk 1 of 1", updates[1][2])
        self.assertIn("recognized speech", updates[2][0])
        self.assertIn("[1.00s", updates[-1][1])
        self.assertIn("Transcription complete", updates[-1][2])
        self.assertTrue(updates[-1][3]["interactive"])
        self.assertFalse(updates[-1][4]["interactive"])

    def test_upload_over_limit_is_rejected_before_loading_model(self):
        with patch.object(app.os.path, "getsize", return_value=app.MAX_AUDIO_SIZE_BYTES + 1):
            with patch.object(app, "get_model") as get_model:
                updates = list(app.transcribe_audio("oversized.wav", model_size="tiny"))

        self.assertIn("limit is 250 MB", updates[0][2])
        get_model.assert_not_called()

    def test_invalid_audio_shows_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "not-audio.wav"
            audio_path.write_text("not audio", encoding="utf-8")
            with patch.object(app, "get_model", return_value=FakeModel()):
                with self.assertLogs("sravana", level="ERROR"):
                    updates = list(
                        app.transcribe_audio(str(audio_path), model_size="tiny")
                    )

        self.assertIn("could not be decoded", updates[-1][2])

    def test_missing_file_error_is_actionable(self):
        with patch.object(app.os.path, "getsize", side_effect=FileNotFoundError):
            updates = list(app.transcribe_audio("missing.wav", model_size="tiny"))

        self.assertIn("no longer available", updates[-1][2])

    def test_transcription_error_preserves_partial_text(self):
        class FailingModel:
            def transcribe(self, audio, **kwargs):
                def segments():
                    yield FakeSegment()
                    raise RuntimeError("decoder failed after a partial result")

                return (
                    segments(),
                    SimpleNamespace(language="te", language_probability=0.98),
                )

        with tempfile.TemporaryDirectory() as directory:
            audio_path = self.make_wav(directory)
            with patch.object(app, "get_model", return_value=FailingModel()):
                with self.assertLogs("sravana", level="ERROR"):
                    updates = list(
                        app.transcribe_audio(str(audio_path), model_size="tiny")
                    )

        self.assertIn("recognized speech", updates[-1][0])
        self.assertIn("unexpected error", updates[-1][2])

    def test_model_cache_keeps_only_last_selected_model(self):
        app.get_model.cache_clear()
        with patch.object(app, "WhisperModel", side_effect=[FakeModel(), FakeModel()]):
            first_model = app.get_model("tiny")
            self.assertIs(app.get_model("tiny"), first_model)
            app.get_model("base")

        self.assertEqual(app.get_model.cache_info().currsize, 1)
        app.get_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
