"""
Telugu Speech-to-Text App
Complete offline-capable app using Faster-Whisper
Supports .m4a, .mp3, .wav, .ogg, .flac etc.
"""

import inspect
import json
import logging
import math
import mimetypes
import os
from functools import lru_cache
from urllib.parse import urlsplit

import av
import ctranslate2
import gradio as gr
import numpy as np
import requests
from faster_whisper import WhisperModel

# -------------------------
# Configuration
# -------------------------
BRAND_NAME = "Sravana Speech Studio"
APP_DESCRIPTION = (
    "Telugu-first speech-to-text with local Whisper or a compatible hosted "
    "transcription model. Powered by Aadhya Labs."
)
APP_TITLE = "Sravana Speech Studio | Telugu Speech-to-Text"
FAQ_ITEMS = [
    (
        "Which languages does Sravana support?",
        "Sravana offers 100 Whisper language codes, including Telugu, English, "
        "Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi, Urdu, Arabic, "
        "Spanish, French, Japanese, Chinese, and Cantonese. Recognition quality "
        "varies by language, model, and audio.",
    ),
    (
        "Does my audio leave my computer?",
        "When you run Sravana on your own computer, audio is transcribed there "
        "and is not sent to an external transcription API in Local Whisper mode. "
        "Hosted provider mode sends audio to the provider you configure; check "
        "its privacy and pricing terms before use.",
    ),
    (
        "What is the maximum audio upload size?",
        "Audio uploads are limited to 250 MiB (262,144,000 bytes).",
    ),
    (
        "Can I stop a long transcription?",
        "Yes. Choose Stop to cancel the transcription. A model inference step "
        "already running may need to finish before cancellation takes effect; "
        "text already returned is retained.",
    ),
    (
        "Which Whisper model should I choose?",
        "Tiny is fastest and uses less memory. Larger models can improve "
        "accuracy, but need more memory and take longer, especially on CPU.",
    ),
]
MODEL_CHOICES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny").strip().lower()
if MODEL_SIZE not in MODEL_CHOICES:
    raise ValueError(
        f"Unsupported WHISPER_MODEL {MODEL_SIZE!r}. "
        f"Choose one of: {', '.join(sorted(MODEL_CHOICES))}."
    )

CHUNK_DURATION_SECONDS = 60
SAMPLE_RATE = 16000
MAX_AUDIO_SIZE_BYTES = 250 * 1024 * 1024
logger = logging.getLogger("sravana")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger.setLevel(logging.INFO)

LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Myanmar", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai",
    "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "yi": "Yiddish", "yo": "Yoruba", "zh": "Chinese", "yue": "Cantonese",
}
SUPPORTED_LANGUAGE_CHOICES = [
    (f"{LANGUAGE_NAMES[code]} ({code})", code)
    for code in sorted(LANGUAGE_NAMES, key=lambda item: LANGUAGE_NAMES[item])
]
LANGUAGE_COUNT = len(SUPPORTED_LANGUAGE_CHOICES)


class AudioInputError(Exception):
    """Raised when an uploaded file does not contain usable audio."""


class ProviderConfigurationError(Exception):
    """Raised when a hosted transcription provider is not configured correctly."""


class ProviderRequestError(Exception):
    """Raised when a hosted transcription provider cannot complete a request."""


DEFAULT_CPU_THREADS = min(4, max(1, (os.cpu_count() or 2) // 2))
CPU_THREADS = max(
    1,
    int(os.getenv("WHISPER_CPU_THREADS", str(DEFAULT_CPU_THREADS))),
)
CUDA_COMPUTE_TYPES = (
    ctranslate2.get_supported_compute_types("cuda")
    if ctranslate2.get_cuda_device_count()
    else set()
)
if "float16" in CUDA_COMPUTE_TYPES:
    DEVICE, COMPUTE_TYPE = "cuda", "float16"
elif "int8_float16" in CUDA_COMPUTE_TYPES:
    DEVICE, COMPUTE_TYPE = "cuda", "int8_float16"
else:
    DEVICE, COMPUTE_TYPE = "cpu", "int8"
TEXTBOX_COPY_BUTTON = (
    {"buttons": ["copy"]}
    if "buttons" in inspect.signature(gr.Textbox).parameters
    else {"show_copy_button": True}
)

@lru_cache(maxsize=1)
def get_model(model_size):
    """Load the selected model and keep only one model resident in memory."""
    logger.info(
        f"Loading Faster-Whisper model '{model_size}' on {DEVICE} "
        f"({COMPUTE_TYPE}, {CPU_THREADS if DEVICE == 'cpu' else 'GPU'} threads)..."
    )
    loaded_model = WhisperModel(
        model_size,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS if DEVICE == "cpu" else 0,
        num_workers=1,
    )
    logger.info("Model %r loaded successfully.", model_size)
    return loaded_model


def ui_update(text, timestamped, status, running=False):
    return (
        text,
        timestamped,
        status,
        gr.update(interactive=not running),
        gr.update(interactive=running),
    )


def format_audio_time(seconds):
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:02}:{minutes:02}:{seconds:02}"
        if hours
        else f"{minutes:02}:{seconds:02}"
    )


def describe_transcription_error(error, phase, model_size):
    if isinstance(error, AudioInputError):
        return str(error)
    if isinstance(error, (ProviderConfigurationError, ProviderRequestError)):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return "The uploaded audio is no longer available. Upload it again and retry."
    if isinstance(error, PermissionError):
        return "The app cannot read this audio file. Check file permissions, then upload it again."
    if isinstance(error, MemoryError):
        return (
            f"There is not enough memory to process this audio with the {model_size} model. "
            "Choose a smaller model or close other applications."
        )
    if isinstance(error, av.error.FFmpegError):
        return (
            "The audio could not be decoded. Try a supported audio file such as MP3, "
            "WAV, M4A, FLAC, or OGG."
        )
    if isinstance(error, (ConnectionError, TimeoutError)) or phase == "loading the model":
        return (
            f"Could not load the {model_size} model. Check your internet connection, "
            "available disk space, and model download, then retry."
        )
    if isinstance(error, OSError):
        return (
            "The app could not access the audio or model files. Check disk space and "
            "file permissions, then retry."
        )
    return (
        f"An unexpected error occurred while {phase}. "
        "Check the app terminal for details, then retry."
    )


def transcribe_with_provider(audio_path, language, api_base_url, api_model, api_key):
    """Send an audio file to an OpenAI-compatible transcription endpoint."""
    api_base_url = api_base_url.strip().rstrip("/")
    api_model = api_model.strip()
    api_key = api_key.strip()

    if not api_key:
        raise ProviderConfigurationError(
            "Enter an API key for the selected hosted provider."
        )
    if not api_model:
        raise ProviderConfigurationError(
            "Enter the transcription model name from your provider."
        )

    parsed_url = urlsplit(api_base_url)
    if (
        parsed_url.scheme not in {"https", "http"}
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ProviderConfigurationError(
            "Enter a valid provider base URL, such as https://api.openai.com/v1."
        )
    if parsed_url.scheme != "https" and parsed_url.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ProviderConfigurationError(
            "Use an HTTPS provider URL to protect your API key and audio."
        )

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": api_model, "response_format": "json"}
    if language is not None:
        data["language"] = language
    endpoint = f"{api_base_url}/audio/transcriptions"
    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"

    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                endpoint,
                headers=headers,
                data=data,
                files={
                    "file": (
                        os.path.basename(audio_path),
                        audio_file,
                        content_type,
                    )
                },
                timeout=(15, 600),
            )
    except requests.RequestException as error:
        raise ProviderRequestError(
            "Could not reach the hosted transcription provider. Check the URL "
            "and your internet connection, then try again."
        ) from error

    if not response.ok:
        try:
            error_body = response.json()
            provider_error = error_body.get("error", error_body)
            detail = (
                provider_error.get("message", "")
                if isinstance(provider_error, dict)
                else str(provider_error)
            )
        except (ValueError, AttributeError):
            detail = ""
        if not detail:
            detail = "Check your API key, model name, and provider limits."
        raise ProviderRequestError(
            f"The hosted provider returned HTTP {response.status_code}: "
            f"{detail[:300]}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise ProviderRequestError(
            "The hosted provider returned an unreadable response. Check that the "
            "selected model supports audio transcription."
        ) from error
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str):
        raise ProviderRequestError(
            "The hosted provider response did not contain transcript text."
        )
    return text.strip()


def iter_audio_chunks(audio_path, chunk_duration=CHUNK_DURATION_SECONDS):
    """Decode audio incrementally and yield 16 kHz mono float chunks with offsets."""
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    buffered_frames = []
    buffered_samples = 0
    total_samples = 0

    with av.open(str(audio_path)) as container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            raise AudioInputError(
                "No audio track was found. Choose an audio recording and try again."
            )

        duration = (
            float(audio_stream.duration * audio_stream.time_base)
            if audio_stream.duration is not None and audio_stream.time_base is not None
            else (
                float(container.duration / av.time_base)
                if container.duration is not None
                else None
            )
        )
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        for frame in container.decode(audio=audio_stream.index):
            for resampled_frame in resampler.resample(frame):
                samples = resampled_frame.to_ndarray().reshape(-1)
                position = 0
                while position < len(samples):
                    take = min(chunk_samples - buffered_samples, len(samples) - position)
                    buffered_frames.append(samples[position : position + take])
                    buffered_samples += take
                    position += take

                    if buffered_samples == chunk_samples:
                        chunk_start = total_samples / SAMPLE_RATE
                        audio = np.concatenate(buffered_frames).astype(np.float32) / 32768.0
                        yield chunk_start, audio, duration
                        total_samples += buffered_samples
                        buffered_frames = []
                        buffered_samples = 0

        for resampled_frame in resampler.resample(None):
            samples = resampled_frame.to_ndarray().reshape(-1)
            buffered_frames.append(samples)
            buffered_samples += len(samples)

    if buffered_samples:
        chunk_start = total_samples / SAMPLE_RATE
        audio = np.concatenate(buffered_frames).astype(np.float32) / 32768.0
        yield chunk_start, audio, duration


def transcribe_audio(
    audio_path,
    language="te",
    beam_size=1,
    model_size=MODEL_SIZE,
    provider="local",
    api_base_url="https://api.openai.com/v1",
    api_model="whisper-1",
    api_key="",
):
    """Transcribe audio in chunks, yielding partial text and progress."""
    if audio_path is None:
        yield ui_update("", "", "Choose an audio file before starting transcription.")
        return

    phase = "checking the uploaded file"
    full_text = []
    timestamped = []
    detected_language = None
    language_probability = None
    try:
        audio_size = os.path.getsize(audio_path)
        if audio_size == 0:
            yield ui_update(
                "",
                "",
                "This file is empty. Choose a valid audio recording and try again.",
            )
            return

        if language is not None and language not in LANGUAGE_NAMES:
            yield ui_update(
                "",
                "",
                "Choose a supported language or select Auto detect.",
            )
            return

        if not isinstance(beam_size, int) or not 1 <= beam_size <= 10:
            yield ui_update("", "", "Choose a beam size between 1 and 10.")
            return

        if audio_size > MAX_AUDIO_SIZE_BYTES:
            size_mb = audio_size / (1024 * 1024)
            limit_mb = MAX_AUDIO_SIZE_BYTES // (1024 * 1024)
            yield ui_update(
                "",
                "",
                f"This file is {size_mb:.1f} MB; the limit is {limit_mb} MB. "
                "Choose a smaller file or compress the audio, then upload it again.",
            )
            return

        if provider not in {"local", "openai-compatible"}:
            yield ui_update("", "", "Choose a supported transcription provider.")
            return

        if provider == "local" and model_size not in MODEL_CHOICES:
            logger.warning("Rejected unsupported Whisper model selection: %r", model_size)
            yield ui_update(
                "",
                "",
                "That model choice is not supported. Select a model from the list and retry.",
            )
            return

        if provider == "openai-compatible":
            phase = "contacting the hosted transcription provider"
            yield ui_update(
                "",
                "",
                "☁️ Sending your audio to the selected provider for transcription...",
                running=True,
            )
            text = transcribe_with_provider(
                audio_path,
                language,
                api_base_url,
                api_model,
                api_key,
            )
            yield ui_update(
                text,
                (
                    "Timestamps are not provided by the selected hosted model.\n\n"
                    + text
                    if text
                    else ""
                ),
                (
                    "✅ Hosted transcription complete"
                    if text
                    else "⚠️ The hosted model returned no speech."
                ),
            )
            return

        chunk_count = 0
        phase = "loading the model"
        yield ui_update(
            "",
            "",
            f"⏳ Preparing your audio and loading the **{model_size}** model if needed...",
            running=True,
        )
        selected_model = get_model(model_size)

        phase = "decoding the audio"
        for chunk_index, (chunk_start, audio_chunk, duration) in enumerate(
            iter_audio_chunks(audio_path),
            start=1,
        ):
            chunk_count = chunk_index
            total_chunks = (
                max(1, math.ceil(duration / CHUNK_DURATION_SECONDS))
                if duration
                else None
            )
            chunk_description = (
                f"chunk {chunk_index} of {total_chunks}"
                if total_chunks
                else f"chunk {chunk_index}"
            )
            header = (
                f"Detected language: {detected_language} "
                f"(confidence: {language_probability:.2%})\n"
                f"{'=' * 50}\n\n"
                if detected_language
                else ""
            )
            chunk_end = chunk_start + len(audio_chunk) / SAMPLE_RATE
            audio_range = (
                f"{format_audio_time(chunk_start)}–{format_audio_time(chunk_end)}"
            )
            yield ui_update(
                header + " ".join(full_text),
                header + "\n".join(timestamped),
                f"🎧 Processing {chunk_description} · audio {audio_range}",
                running=True,
            )

            phase = f"transcribing {chunk_description}"
            segments, info = selected_model.transcribe(
                audio_chunk,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            detected_language = detected_language or info.language
            language_probability = language_probability or info.language_probability

            for segment in segments:
                text = segment.text.strip()
                if text:
                    full_text.append(text)
                    timestamped.append(
                        f"[{chunk_start + segment.start:.2f}s → "
                        f"{chunk_start + segment.end:.2f}s] {text}"
                    )
                    header = (
                        f"Detected language: {detected_language} "
                        f"(confidence: {language_probability:.2%})\n"
                        f"{'=' * 50}\n\n"
                    )
                    segment_percent = (
                        min(int((chunk_start + segment.end) / duration * 100), 99)
                        if duration
                        else None
                    )
                    segment_time = format_audio_time(chunk_start + segment.end)
                    yield ui_update(
                        header + " ".join(full_text),
                        header + "\n".join(timestamped),
                        f"✍️ Recognizing speech in {chunk_description}"
                        + (
                            f" · audio position {segment_percent}% ({segment_time})"
                            if segment_percent is not None
                            else f" · audio position {segment_time}"
                        ),
                        running=True,
                    )

            finished_percent = (
                min(
                    int(
                        (chunk_start + len(audio_chunk) / SAMPLE_RATE)
                        / duration
                        * 100
                    ),
                    99,
                )
                if duration
                else None
            )
            phase = "decoding the audio"
            yield ui_update(
                header + " ".join(full_text),
                header + "\n".join(timestamped),
                f"✅ Finished {chunk_description}"
                + (
                    f" · {finished_percent}% of audio processed"
                    if finished_percent is not None
                    else ""
                ),
                running=True,
            )

        if chunk_count == 0:
            yield ui_update("", "", "No audio could be decoded from this file.")
            return

        header = (
            f"Detected language: {detected_language} "
            f"(confidence: {language_probability:.2%})\n"
            f"{'=' * 50}\n\n"
        )
        yield ui_update(
            header + " ".join(full_text),
            header + "\n".join(timestamped),
            (
                f"✅ Transcription complete · {chunk_count} chunk(s) processed"
                if full_text
                else
                f"⚠️ Processed {chunk_count} chunk(s), but no speech was recognized. "
                "Check the selected language and audio quality."
            ),
        )

    except Exception as e:
        filename = os.path.basename(str(audio_path))
        logger.exception("Transcription failed while %s for %s", phase, filename)
        partial_header = (
            f"Detected language: {detected_language} "
            f"(confidence: {language_probability:.2%})\n"
            f"{'=' * 50}\n\n"
            if detected_language and language_probability is not None
            else ""
        )
        yield ui_update(
            partial_header + " ".join(full_text),
            partial_header + "\n".join(timestamped),
            f"⚠️ {describe_transcription_error(e, phase, model_size)}",
        )


# -------------------------
# Gradio UI
# -------------------------
title = f"🎙️ {BRAND_NAME}"
description = APP_DESCRIPTION

STRUCTURED_DATA = json.dumps(
    {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "SoftwareApplication",
                "name": BRAND_NAME,
                "applicationCategory": "MultimediaApplication",
                "operatingSystem": "Windows, macOS, Linux",
                "description": APP_DESCRIPTION,
                "featureList": [
                    "Telugu speech-to-text transcription",
                    f"Transcription in {LANGUAGE_COUNT} Whisper-supported languages",
                    "Local and compatible hosted transcription providers",
                    "Live transcription progress and timestamps",
                    "Audio upload limit of 250 MB",
                ],
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": question,
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": answer,
                        },
                    }
                    for question, answer in FAQ_ITEMS
                ],
            },
        ],
    },
    ensure_ascii=False,
)

PAGE_HEAD = f"""
<meta name="description" content="{APP_DESCRIPTION}">
<meta name="keywords" content="Telugu speech to text, Telugu audio transcription, Telugu transcription, multilingual speech recognition, offline Whisper transcription, audio to text">
<meta name="application-name" content="{BRAND_NAME}">
<meta name="theme-color" content="#174e55">
<meta property="og:type" content="website">
<meta property="og:title" content="{APP_TITLE}">
<meta property="og:description" content="{APP_DESCRIPTION}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{APP_TITLE}">
<meta name="twitter:description" content="{APP_DESCRIPTION}">
<script type="application/ld+json">
{STRUCTURED_DATA}
</script>
"""

APP_CSS = """
.gradio-container {max-width: 1440px !important; margin: 0 auto !important;}
.app-hero {
    padding: 26px 30px !important;
    border: 0 !important;
    border-radius: 22px !important;
    background: linear-gradient(120deg, #102d3a 0%, #174e55 58%, #236a65 100%);
    color: #f5fbfa !important;
    margin-bottom: 20px !important;
    box-shadow: 0 12px 30px rgba(16, 45, 58, 0.14);
}
.app-hero > div,
.app-hero .app-hero {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    border-radius: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.app-hero h1 {margin: 0 0 8px 0 !important; font-size: 2rem !important; color: #f5fbfa !important;}
.app-hero p {margin: 0 !important; color: #d3e6e2 !important;}
.language-banner {
    text-align: center;
    color: #17635d;
    font-weight: 650;
    letter-spacing: 0.01em;
    padding: 2px 8px 14px;
}
.app-panel {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    background: var(--background-fill-primary);
    box-shadow: 0 5px 18px rgba(25, 45, 55, 0.055);
}
.app-status {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 14px !important;
    padding: 10px 15px !important;
    background: #f2f8f7;
    min-height: 48px;
}
#transcribe-button button {
    min-height: 52px;
    border-radius: 13px;
    font-weight: 700;
}
#stop-button button {min-height: 52px; border-radius: 13px; font-weight: 650;}
.app-footer {text-align: center; color: var(--body-text-color-subdued); padding: 10px;}
@media (max-width: 800px) {
    .app-hero {padding: 20px !important;}
    .app-panel {padding: 15px !important;}
}
"""

with gr.Blocks(title=title) as demo:
    gr.Markdown(
        f"# {title}\n\n{description}",
        elem_classes=["app-hero"],
    )
    gr.Markdown(
        "**Telugu · English · Hindi · Tamil · Kannada · Malayalam · "
        f"and {LANGUAGE_COUNT - 6} more supported languages**",
        elem_classes=["language-banner"],
    )

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["app-panel"]):
            gr.Markdown("### 1 · Choose audio")
            audio_input = gr.Audio(
                label="Drop a recording here or browse",
                type="filepath",
                sources=["upload", "microphone"],
            )
            gr.Markdown(
                f"**Maximum file size: {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MB**"
            )

            gr.Markdown("### 2 · Transcription settings")
            provider_selector = gr.Dropdown(
                choices=[
                    ("Local Whisper · private and offline", "local"),
                    ("Hosted API · OpenAI-compatible", "openai-compatible"),
                ],
                value="local",
                label="Transcription provider",
                info="Local Whisper is the simplest private option. Hosted APIs may have usage fees.",
            )
            with gr.Group(visible=True) as local_settings:
                model_selector = gr.Dropdown(
                    choices=[
                        ("Tiny · fastest, lower accuracy", "tiny"),
                        ("Base · fast, improved accuracy", "base"),
                        ("Small · balanced", "small"),
                        ("Medium · higher accuracy, slower on CPU", "medium"),
                        ("Large v2 · highest accuracy, slowest", "large-v2"),
                        ("Large v3 · highest accuracy, slowest", "large-v3"),
                    ],
                    value=MODEL_SIZE,
                    label="Whisper model",
                    info="A model is downloaded the first time you use it. Only one model is kept in memory at a time.",
                )
                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Accuracy vs. speed",
                    info="1 is fastest; higher values may improve accuracy but take longer.",
                )
                gr.Markdown(
                    f"**Compute:** `{DEVICE}` · `{COMPUTE_TYPE}` · "
                    f"`{CPU_THREADS if DEVICE == 'cpu' else 'GPU'}` threads"
                )
            with gr.Group(visible=False) as hosted_settings:
                gr.Markdown(
                    "Connect any provider that supports the OpenAI-compatible "
                    "`/audio/transcriptions` API. For example, OpenAI uses "
                    "`https://api.openai.com/v1` and model `whisper-1`; Groq uses "
                    "`https://api.groq.com/openai/v1` and model `whisper-large-v3`."
                )
                api_base_url = gr.Textbox(
                    value="https://api.openai.com/v1",
                    label="Provider base URL",
                    placeholder="https://api.openai.com/v1",
                )
                api_model = gr.Textbox(
                    value="whisper-1",
                    label="Transcription model",
                    placeholder="whisper-1",
                )
                api_key = gr.Textbox(
                    label="API key",
                    placeholder="Paste your provider API key",
                    type="password",
                )
                gr.Markdown(
                    "Your key is used for this request only and is not saved by "
                    "Sravana. Hosted mode sends the full audio file to that provider; "
                    "its privacy policy, file limits, and charges apply."
                )
            language = gr.Dropdown(
                choices=[("Auto detect", None), *SUPPORTED_LANGUAGE_CHOICES],
                value="te",
                label="Audio language",
                info=f"Select from {LANGUAGE_COUNT} Whisper-supported languages or let the app detect it.",
            )
            with gr.Row():
                transcribe_btn = gr.Button(
                    "Transcribe audio",
                    variant="primary",
                    size="lg",
                    elem_id="transcribe-button",
                    scale=3,
                )
                stop_btn = gr.Button(
                    "Stop",
                    variant="stop",
                    size="lg",
                    elem_id="stop-button",
                    interactive=False,
                    scale=1,
                )
            progress_output = gr.Markdown(
                "Ready when you are — upload audio to get started.",
                elem_classes=["app-status"],
            )

        with gr.Column(scale=6, elem_classes=["app-panel"]):
            gr.Markdown("### 3 · Your transcript")
            with gr.Tabs():
                with gr.Tab("Plain text"):
                    text_output = gr.Textbox(
                        label="Live transcription",
                        lines=20,
                        max_lines=30,
                        placeholder="Your recognized speech will appear here as each segment is transcribed.",
                        **TEXTBOX_COPY_BUTTON,
                    )
                with gr.Tab("Timestamps"):
                    timestamp_output = gr.Textbox(
                        label="Transcript with timestamps",
                        lines=20,
                        max_lines=30,
                        placeholder="Timestamped speech segments will appear here.",
                        **TEXTBOX_COPY_BUTTON,
                    )

    transcribe_event = transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[
            audio_input,
            language,
            beam_size,
            model_selector,
            provider_selector,
            api_base_url,
            api_model,
            api_key,
        ],
        outputs=[text_output, timestamp_output, progress_output, transcribe_btn, stop_btn],
        concurrency_limit=1,
        show_progress="hidden",
    )

    def toggle_provider_settings(selected_provider):
        return (
            gr.update(visible=selected_provider == "local"),
            gr.update(visible=selected_provider == "openai-compatible"),
        )

    provider_selector.change(
        fn=toggle_provider_settings,
        inputs=provider_selector,
        outputs=[local_settings, hosted_settings],
        queue=False,
    )

    def stop_transcription():
        return (
                "⏹️ Stop requested. Text already produced is preserved; the current "
                "model step may finish before the job stops.",
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

    stop_btn.click(
        fn=stop_transcription,
        outputs=[progress_output, transcribe_btn, stop_btn],
        cancels=[transcribe_event],
        queue=False,
    )

    with gr.Accordion("Language support and frequently asked questions", open=False):
        gr.Markdown(
            "\n\n".join(f"### {question}\n\n{answer}" for question, answer in FAQ_ITEMS)
        )

    gr.Markdown(
        f"Powered by Aadhya Labs · Local {CHUNK_DURATION_SECONDS}-second chunks · "
        f"{LANGUAGE_COUNT} supported language codes",
        elem_classes=["app-footer"],
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=APP_CSS,
        head=PAGE_HEAD,
        footer_links=[],
        max_file_size=MAX_AUDIO_SIZE_BYTES,
    )
