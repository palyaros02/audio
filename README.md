# audio

Python 3.12 CLI for audio transcription and text-to-speech through OpenRouter.

Old transcription behavior is now available through the `audio transcribe` subcommand:

```bash
audio transcribe /path/to/file
```

Text-to-speech is available through `audio tts` and uses `google/gemini-3.1-flash-tts-preview` by default:

```bash
audio tts "Hello from Gemini TTS" --out speech.ogg
```

## Setup

Install dependencies with uv:

```bash
uv sync
```

Put your OpenRouter key into `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

TTS conversion requires `ffmpeg`, because Gemini TTS output is requested as raw PCM and then converted locally to `ogg` or `mp3`.

For local development, run through uv:

```bash
uv run audio transcribe /path/to/audio.wav
uv run audio tts "Hello" --out speech.ogg
```

To expose `audio` as a regular command in your current Python environment:

```bash
uv tool install .
```

## Transcription

```bash
audio transcribe [OPTIONS] MEDIA_PATH
```

Options:

```text
--model TEXT      OpenRouter model to use. Default: google/gemini-3.1-flash-lite
--prompt TEXT     Instruction sent with the audio. Default: Generate a transcript of the speech.
--out TEXT        stdout or output file path. Default: stdout
--timeout FLOAT   HTTP timeout in seconds. Default: 120
```

Examples:

```bash
audio transcribe ./voice-message.mp3
audio transcribe ./voice-message.mp3 --out transcript.txt
audio transcribe ./voice-message.mp3 > transcript.txt
audio transcribe ./voice-message.mp3 --prompt "Transcribe this speech verbatim. Keep the original language."
```

For OpenClaw, use:

```json
["audio", "transcribe", "{{MediaPath}}"]
```

Supported transcription file extensions: `aac`, `aiff`, `flac`, `m4a`, `mp3`, `ogg`, `opus`, `pcm16`, `pcm24`, `wav`, `webm`.

## Text-To-Speech

```bash
audio tts [OPTIONS] [TEXT ...]
```

If `TEXT` is omitted, `audio tts` reads text from stdin.

Options:

```text
--voices          Print supported voices and exit
--models          Print available OpenRouter TTS models and exit
--model TEXT      OpenRouter TTS model. Default: google/gemini-3.1-flash-tts-preview
--voice TEXT      Voice to use. Default: Zephyr for Gemini, alloy for OpenAI TTS
--out TEXT        Output audio path. Default: speech.ogg
--format FORMAT   Output format: ogg or mp3. Inferred from --out when omitted;
                  when set, --out extension is adjusted to match
--timeout FLOAT   HTTP timeout in seconds. Default: 120
```

Examples:

```bash
audio tts "Привет, это голосовой ответ" --out answer.ogg
audio tts "Hello" --voice Puck --out answer.mp3
audio tts --model openai/gpt-4o-mini-tts-2025-12-15 "Hello" --voice nova --out answer.ogg
printf "Long text" | audio tts --out narration.ogg
audio tts --voices
audio tts --models
```

`audio tts --models` queries OpenRouter for models with `output_modalities=speech` and prints all currently available TTS model IDs.

`openai/gpt-4o-audio-preview` is not a TTS model on OpenRouter's `/audio/speech` endpoint. For OpenAI TTS, use `openai/gpt-4o-mini-tts-2025-12-15`.

Voices:

```text
Zephyr
Puck
Charon
Kore
Fenrir
Leda
Orus
Aoede
Callirrhoe
Autonoe
Enceladus
Iapetus
Umbriel
Algieba
Despina
Erinome
Algenib
```

TTS requests `pcm` from OpenRouter first. The CLI then converts the 24 kHz, 16-bit, mono PCM stream to `ogg` by default, or to `mp3` when requested. If a provider rejects PCM and only supports MP3, the CLI automatically retries with `response_format=mp3` and converts that file when needed.

## Notes

The CLI uses the OpenAI Python SDK pointed at `https://openrouter.ai/api/v1`.

Transcription sends base64-encoded local audio to the chat completions API using an `input_audio` message part, which keeps custom transcription prompts available.
