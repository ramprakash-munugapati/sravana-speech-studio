# Sravana Speech Studio — Telugu Speech-to-Text

**Sravana** is a Telugu-first, multilingual audio transcription app powered by
[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and Aadhya Labs.
Use local Whisper or connect a hosted, OpenAI-compatible transcription API.
Transcribe recordings in a Gradio web app and copy plain text or timestamped
local results.

Search: Telugu speech to text · Telugu audio transcription · speech recognition
for Telugu · multilingual audio to text · offline Whisper transcription.

## What languages does Sravana support?

Sravana offers **100 Whisper language codes** in its searchable language
selector. These include Telugu, English, Hindi, Tamil, Kannada, Malayalam,
Bengali, Marathi, Urdu, Arabic, Spanish, French, German, Japanese, Chinese,
Cantonese, and many more. Select **Auto detect** when you do not know the
recording's language. Recognition accuracy varies with the model, recording
quality, accents, and background noise.

## Features

- Telugu-first speech recognition with selectable Whisper models
- Optional OpenAI-compatible hosted speech-to-text APIs, including OpenAI and Groq
- Multilingual speech-to-text, including English, Hindi, Tamil, Kannada, and Malayalam
- Local Gradio web UI and command-line transcription tool
- Processes long recordings in 60-second chunks and shows partial transcription and progress
- Live transcription status with separate plain-text and timestamped results
- Stop long-running transcriptions while retaining text already produced
- Audio uploads limited to 250 MiB
- Tiny model and beam size 1 defaults for CPU speed; selectable larger models for accuracy
- CPU int8 inference, with CUDA selected when a compatible compute type is available
- After the selected model is downloaded, local transcription can run offline

## Quick Start

### Requirements

- Python 3.9 or newer
- Internet access the first time you download a Whisper model
- Enough memory and disk space for the selected model and audio

Faster-Whisper decodes audio using its bundled PyAV libraries; a separate FFmpeg
installation is generally not required.

### Install and run on Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The first transcription
downloads the selected model if it is not cached. Larger models need more
memory and take longer, particularly on CPU.

### Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

### Using the web app

1. Upload an audio file (up to 250 MiB) or record from your microphone.
2. Choose **Local Whisper** for on-device transcription or **Hosted API** to
   connect an OpenAI-compatible transcription provider.
3. For local transcription, select the spoken language and pick a Whisper model
   and beam size. Tiny and beam size 1 are fastest; larger models or beam sizes
   may improve accuracy and take longer.
4. For a hosted provider, enter its base URL, transcription model, and API key.
   The key is not saved in Sravana's app settings.
5. Choose **Transcribe audio**. Local mode shows partial text as chunks finish;
   hosted mode sends the full recording to the provider.
6. Choose **Stop** to cancel. Text already produced remains available.

#### Hosted provider examples

| Provider | Base URL | Example transcription model |
|----------|----------|-----------------------------|
| OpenAI | `https://api.openai.com/v1` | `whisper-1` |
| Groq | `https://api.groq.com/openai/v1` | `whisper-large-v3` |

Use a provider that supports the OpenAI-compatible
`POST /audio/transcriptions` endpoint. Model names, availability, pricing, file
limits, and supported formats vary by provider. Sravana sends the full audio
recording directly to the configured endpoint over HTTPS; the provider's
privacy and data-retention terms apply. Local Whisper remains available and
does not upload recordings to a transcription API.

Supported models: `tiny`, `base`, `small`, `medium`, `large-v2`, and `large-v3`.
Only the most recently selected model stays in memory. Override the default CPU
thread count with `WHISPER_CPU_THREADS` if needed.

### 3. Use the Command Line

```bash
# Basic usage
python transcribe_cli.py your_audio.m4a

# With timestamps
python transcribe_cli.py your_audio.m4a --timestamps

# Save to file
python transcribe_cli.py your_audio.m4a -o output.txt

# Use larger model for better accuracy
python transcribe_cli.py your_audio.m4a --model large-v3

# Auto language detection
python transcribe_cli.py your_audio.m4a --language auto
```

## Frequently asked questions

### Can I transcribe Telugu audio to text?

Yes. Choose Telugu in Sravana's language selector to transcribe Telugu
recordings into text. Select a larger Whisper model for potentially better
accuracy when your computer has enough memory.

### Which languages does Sravana support?

The app exposes 100 language codes supported by Whisper, including Telugu,
English, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi, Urdu, Arabic,
Spanish, French, Japanese, Chinese, and Cantonese. Recognition quality varies
by language, model, and audio.

### Does Sravana upload my audio to an online transcription API?

In **Local Whisper** mode, audio is transcribed on the machine running the app.
In **Hosted API** mode, the recording is sent to the provider you configured.
Review its privacy and data-retention terms before using hosted transcription.

### What is the maximum audio file size?

The Gradio web app accepts audio uploads up to **250 MiB**
(262,144,000 bytes). Split or compress a larger recording before uploading.

### Can I stop a transcription already in progress?

Yes. Use **Stop** to cancel it. A model inference step already running may need
to finish before cancellation takes effect; text already returned is retained.

## Which Whisper model should I choose?

| Model     | Size   | Speed     | Accuracy      | Recommended for          |
|-----------|--------|-----------|---------------|--------------------------|
| **tiny**  | ~75 MB  | **Very fast** | Lower | Fastest inference and lower-memory systems |
| base      | ~150 MB | Fast          | Medium | Speed and accuracy balance |
| small     | ~500 MB | Medium        | Good | General-purpose results |
| medium    | ~1.5 GB | Slower on CPU | Very good | Higher accuracy |
| large-v2  | ~3 GB   | Slowest on CPU | Best | Highest accuracy when hardware allows |
| large-v3  | ~3 GB   | Slowest on CPU | Best | Highest accuracy when hardware allows |

Model sizes are approximate; disk, RAM, audio duration, hardware, language, and
recording quality affect speed and results.

## Is my audio sent to a transcription service?

When using Local Whisper on your own computer, transcription runs locally; an
internet connection may be needed to download a model the first time. Hosted
API mode sends audio to the configured provider. Do not expose the local
Gradio server to an untrusted network without adding appropriate access
controls.

## What file types and sizes are supported?

The web app accepts audio formats PyAV can decode, including common MP3, WAV,
M4A, FLAC, and OGG recordings. The maximum upload size is **250 MiB**
(262,144,000 bytes). If an upload is too large, compress the audio or split it
into smaller files.

## Troubleshooting

- **Model download fails:** Check your internet connection and available disk
  space, then retry. Once downloaded, the model is cached for offline use.
- **Not enough memory:** Choose a smaller Whisper model and close other
  applications before retrying.
- **Audio cannot be decoded:** Try converting or re-exporting the recording as
  MP3 or WAV.
- **No speech recognized:** Check that the selected language matches the audio
  and that the recording is audible.
- **Transcription is slow:** Use the tiny model and beam size 1, or split long
  recordings into shorter files.

## Project Structure

```
telugu-stt-app/
├── app.py              # Sravana Gradio web app
├── transcribe_cli.py   # Command-line transcription
├── requirements.txt
├── .gitignore
├── tests/
│   └── test_app.py     # Web app regression tests
└── README.md
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

## How is Telugu transcription accuracy improved?

Use the correct spoken-language setting, select a larger model when memory
allows, and record clear audio with minimal background noise. Tiny is optimized
for speed, not maximum accuracy. Review names and uncommon words in any
machine-generated transcript.

## License

Powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and
Whisper speech recognition models. Built by **Aadhya Labs**. Review the
project, model, and hosted provider licenses and terms before redistributing
or deploying.
