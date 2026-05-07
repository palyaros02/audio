from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audio


class TtsOutputPathTests(unittest.TestCase):
    def test_requested_format_replaces_mismatched_output_extension(self) -> None:
        with (
            patch("audio.synthesize_audio_to_file", return_value=audio.TTS_PCM_FORMAT),
            patch("audio.convert_audio") as convert_audio,
        ):
            output_path = audio.tts(["hello"], audio.TTS_MODEL, None, "test.mp3", "ogg", 1.0)

        self.assertEqual(output_path, Path("test.ogg"))
        self.assertEqual(convert_audio.call_args.args[2], Path("test.ogg"))
        self.assertEqual(convert_audio.call_args.args[3], "ogg")

    def test_requested_format_adds_missing_output_extension(self) -> None:
        output_path = audio.resolve_tts_output_path(Path("test"), "mp3", "mp3")

        self.assertEqual(output_path, Path("test.mp3"))


if __name__ == "__main__":
    unittest.main()
