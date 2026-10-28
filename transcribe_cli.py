#!/usr/bin/env python3
"""
Telugu Speech-to-Text CLI
Usage:
    python transcribe_cli.py your_audio.m4a
    python transcribe_cli.py your_audio.m4a --model large-v3 --language te
"""

import argparse
import sys
from pathlib import Path
from faster_whisper import WhisperModel
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Telugu (or other) audio to text using Faster-Whisper"
    )
    parser.add_argument("audio", type=str, help="Path to audio file (.m4a, .mp3, .wav, etc.)")
    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "--language", "-l",
        default="te",
        help="Language code (default: te for Telugu). Use 'auto' for detection"
    )
    parser.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="Print timestamps for each segment"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save transcription to a text file"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found → {audio_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading model '{args.model}' on {device}...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    language = None if args.language.lower() == "auto" else args.language

    print(f"Transcribing: {audio_path.name} (language={language or 'auto'})")
    print("-" * 50)

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=args.beam_size,
        vad_filter=True,
    )

    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2%})")
    print("-" * 50)

    full_text = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            full_text.append(text)
            if args.timestamps:
                print(f"[{segment.start:7.2f}s → {segment.end:7.2f}s]  {text}")
            else:
                print(text)

    result = " ".join(full_text)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result, encoding="utf-8")
        print(f"\nSaved to: {out_path}")

    print("\n" + "=" * 50)
    print("FULL TRANSCRIPTION:")
    print(result)


if __name__ == "__main__":
    main()
