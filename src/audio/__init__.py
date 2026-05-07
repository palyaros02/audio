from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TRANSCRIBE_MODEL = "google/gemini-3.1-flash-lite"
TRANSCRIBE_PROMPT = "Generate a transcript of the speech."
TTS_MODEL = "google/gemini-3.1-flash-tts-preview"
TTS_VOICE = "Zephyr"
OPENAI_TTS_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"
OPENAI_TTS_VOICE = "alloy"
TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS = 1
TTS_SAMPLE_FORMAT = "s16le"
TTS_PCM_FORMAT = "pcm"
TTS_MP3_FORMAT = "mp3"

SUPPORTED_TRANSCRIBE_FORMATS = {
    "aac",
    "aiff",
    "flac",
    "m4a",
    "mp3",
    "ogg",
    "opus",
    "pcm16",
    "pcm24",
    "wav",
    "webm",
}
EXTENSION_ALIASES = {
    "aif": "aiff",
    "aifc": "aiff",
    "oga": "ogg",
}
TTS_VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
)
VOICE_BY_LOWERCASE = {voice.lower(): voice for voice in TTS_VOICES}
OPENAI_TTS_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}
UNSUPPORTED_TTS_MODELS = {
    "openai/gpt-4o-audio-preview": (
        "openai/gpt-4o-audio-preview is a chat/audio model on OpenRouter, "
        f"not a /audio/speech TTS model. Use the default {TTS_MODEL} or "
        f"{OPENAI_TTS_MODEL}."
    ),
}


class AudioError(Exception):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audio",
        description="Audio utilities powered by OpenRouter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file",
        description="Transcribe an audio file with OpenRouter.",
    )
    transcribe_parser.add_argument("media_path", type=Path, help="Path to an audio file")
    transcribe_parser.add_argument(
        "--model",
        default=TRANSCRIBE_MODEL,
        help=f"OpenRouter model to use (default: {TRANSCRIBE_MODEL})",
    )
    transcribe_parser.add_argument(
        "--prompt",
        default=TRANSCRIBE_PROMPT,
        help=f"Instruction sent with the audio (default: {TRANSCRIBE_PROMPT!r})",
    )
    transcribe_parser.add_argument(
        "--out",
        default="stdout",
        help="Output destination: stdout or a file path (default: stdout)",
    )
    transcribe_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )

    tts_parser = subparsers.add_parser(
        "tts",
        help="Generate speech from text",
        description="Generate speech from text with Gemini TTS.",
    )
    tts_parser.add_argument("text", nargs="*", help="Text to synthesize. Reads stdin if omitted.")
    tts_parser.add_argument(
        "--voices",
        action="store_true",
        help="Print supported Gemini TTS voices and exit",
    )
    tts_parser.add_argument(
        "--models",
        action="store_true",
        help="Print available OpenRouter TTS models and exit",
    )
    tts_parser.add_argument(
        "--model",
        default=TTS_MODEL,
        help=f"OpenRouter TTS model to use (default: {TTS_MODEL})",
    )
    tts_parser.add_argument(
        "--voice",
        help=f"Voice to use (default: {TTS_VOICE} for Gemini, {OPENAI_TTS_VOICE} for OpenAI TTS)",
    )
    tts_parser.add_argument(
        "--out",
        default="speech.ogg",
        help="Output audio path (default: speech.ogg)",
    )
    tts_parser.add_argument(
        "--format",
        choices=("ogg", "mp3"),
        help="Output format. Inferred from --out when omitted; defaults to ogg.",
    )
    tts_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )

    return parser.parse_args(argv)


def transcribe_audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    suffix = EXTENSION_ALIASES.get(suffix, suffix)
    if suffix not in SUPPORTED_TRANSCRIBE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TRANSCRIBE_FORMATS))
        raise AudioError(
            f"Unsupported audio extension '.{path.suffix.lstrip('.')}'. "
            f"Supported formats: {supported}."
        )
    return suffix


def read_audio(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise AudioError(f"File not found: {path}")
    if not path.is_file():
        raise AudioError(f"Not a file: {path}")

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return data, transcribe_audio_format(path)


def openrouter_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": "https://github.com/personal/audio",
        "X-Title": "audio-cli",
        "X-OpenRouter-Title": "audio-cli",
    }


def openrouter_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise AudioError("OPENROUTER_API_KEY is not set. Add it to .env or export it.")
    return api_key


def openrouter_client(timeout: float) -> OpenAI:
    return OpenAI(
        api_key=openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        default_headers=openrouter_headers(),
        timeout=timeout,
    )


def describe_openai_error(exc: APIError) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code:
        return f"HTTP {status_code}: {exc.message}"
    return exc.message


def extract_transcript(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        text = "".join(parts).strip()
        if text:
            return text

    raise AudioError(f"OpenRouter response did not contain text: {content}")


def transcribe(path: Path, model: str, prompt: str, timeout: float) -> str:
    audio_data, format_ = read_audio(path)
    try:
        response = openrouter_client(timeout).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": format_,
                            },
                        },
                    ],
                }
            ],
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise AudioError(f"OpenRouter request failed: {exc}") from exc
    except APIError as exc:
        raise AudioError(f"OpenRouter request failed with {describe_openai_error(exc)}") from exc

    if not response.choices:
        raise AudioError(f"OpenRouter response did not contain choices: {response}")
    return extract_transcript(response.choices[0].message.content)


def write_text_output(text: str, destination: str) -> None:
    if destination == "stdout":
        print(text)
        return

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def validate_tts_model(model: str) -> None:
    if model in UNSUPPORTED_TTS_MODELS:
        raise AudioError(UNSUPPORTED_TTS_MODELS[model])


def list_tts_models(timeout: float) -> list[str]:
    try:
        payload = openrouter_client(timeout).get("/models?output_modalities=speech", cast_to=object)
    except (APIConnectionError, APITimeoutError) as exc:
        raise AudioError(f"OpenRouter models request failed: {exc}") from exc
    except APIError as exc:
        raise AudioError(f"OpenRouter models request failed with {describe_openai_error(exc)}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise AudioError(f"Unexpected OpenRouter models response: {payload}")

    models = [model.get("id") for model in payload["data"] if isinstance(model, dict)]
    return [model for model in models if isinstance(model, str)]


def normalize_voice(model: str, voice: str | None) -> str:
    if model.startswith("google/gemini"):
        normalized = VOICE_BY_LOWERCASE.get((voice or TTS_VOICE).lower())
        if normalized:
            return normalized
        raise AudioError(f"Unsupported Gemini voice: {voice}. Run 'audio tts --voices' to list voices.")

    if model.startswith("openai/gpt-4o-mini-tts"):
        normalized = (voice or OPENAI_TTS_VOICE).lower()
        if normalized in OPENAI_TTS_VOICES:
            return normalized
        voices = ", ".join(sorted(OPENAI_TTS_VOICES))
        raise AudioError(f"Unsupported OpenAI TTS voice: {voice}. Supported voices: {voices}.")

    if not voice:
        raise AudioError(
            f"Voice is required for custom TTS model {model}. "
            f"Known models: {TTS_MODEL}, {OPENAI_TTS_MODEL}."
        )
    return voice


def read_tts_text(text_parts: list[str]) -> str:
    text = " ".join(text_parts).strip()
    if text:
        return text

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text

    raise AudioError("Text is required. Pass it as an argument or pipe it via stdin.")


def infer_tts_format(output_path: Path, requested_format: str | None) -> str:
    if requested_format:
        return requested_format

    suffix = output_path.suffix.lower().lstrip(".")
    if not suffix:
        return "ogg"
    if suffix in {"ogg", "mp3"}:
        return suffix
    raise AudioError("TTS output format must be ogg or mp3. Use --format to choose one.")


def tts_requires_mp3(exc: APIError) -> bool:
    message = exc.message.lower()
    return "response_format" in message and "mp3" in message and "pcm" in message


def synthesize_audio_to_file(
    text: str,
    model: str,
    voice: str,
    response_format: str,
    timeout: float,
    output_path: Path,
) -> str:
    try:
        with openrouter_client(timeout).audio.speech.with_streaming_response.create(
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,
        ) as response:
            response.stream_to_file(output_path)
    except (APIConnectionError, APITimeoutError) as exc:
        raise AudioError(f"OpenRouter TTS request failed: {exc}") from exc
    except APIError as exc:
        if response_format == TTS_PCM_FORMAT and tts_requires_mp3(exc):
            return synthesize_audio_to_file(text, model, voice, TTS_MP3_FORMAT, timeout, output_path)
        raise AudioError(f"OpenRouter TTS request failed with {describe_openai_error(exc)}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioError("OpenRouter TTS response was empty.")
    return response_format


def convert_audio(input_path: Path, input_format: str, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if input_format == TTS_PCM_FORMAT:
        command.extend([
            "-f",
            TTS_SAMPLE_FORMAT,
            "-ar",
            str(TTS_SAMPLE_RATE),
            "-ac",
            str(TTS_CHANNELS),
        ])
    command.extend(["-i", str(input_path)])

    if output_format == "ogg":
        command.extend(["-c:a", "libopus"])
    elif output_format == "mp3":
        command.extend(["-c:a", "libmp3lame"])
    else:
        raise AudioError(f"Unsupported TTS output format: {output_format}")
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioError("ffmpeg is required to convert Gemini PCM output to ogg/mp3.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "unknown ffmpeg error"
        raise AudioError(f"ffmpeg conversion failed: {stderr}") from exc


def tts(text_parts: list[str], model: str, voice: str | None, output: str, output_format: str | None, timeout: float) -> Path:
    output_path = Path(output)
    validate_tts_model(model)
    normalized_voice = normalize_voice(model, voice)
    resolved_format = infer_tts_format(output_path, output_format)
    with tempfile.TemporaryDirectory(prefix="audio-tts-") as temp_dir:
        generated_path = Path(temp_dir) / "speech.audio"
        generated_format = synthesize_audio_to_file(
            read_tts_text(text_parts),
            model,
            normalized_voice,
            TTS_PCM_FORMAT,
            timeout,
            generated_path,
        )
        convert_audio(generated_path, generated_format, output_path, resolved_format)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "transcribe":
            text = transcribe(args.media_path, args.model, args.prompt, args.timeout)
            write_text_output(text, args.out)
            return 0

        if args.command == "tts":
            if args.voices:
                print("\n".join(TTS_VOICES))
                return 0
            if args.models:
                print("\n".join(list_tts_models(args.timeout)))
                return 0
            output_path = tts(args.text, args.model, args.voice, args.out, args.format, args.timeout)
            print(output_path)
            return 0
    except AudioError as exc:
        print(f"audio: error: {exc}", file=sys.stderr)
        return 1

    print(f"audio: error: unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
