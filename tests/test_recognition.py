import tempfile
import unittest
import wave
from pathlib import Path
from typing import ClassVar

import numpy as np

from sound_analyzer import RecognitionError, UnsupportedPhoneError
from sound_analyzer.infrastructure.recognition import (
    TARGET_SAMPLE_RATE,
    _mask_logits,
    _read_audio,
)


class _Tokenizer:
    all_special_ids: ClassVar[list[int]] = [0]

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<pad>": 0, "a": 1, "ɡ": 2, "ʃ": 3}


class RecognitionTests(unittest.TestCase):
    def test_reads_stereo_pcm_and_resamples_to_model_rate(self) -> None:
        sample_rate = 8_000
        samples = np.zeros((sample_rate, 2), dtype=np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())

            audio = _read_audio(Path(temporary.name))

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), TARGET_SAMPLE_RATE)

    def test_rejects_non_pcm16_wav(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(TARGET_SAMPLE_RATE)
                wav_file.writeframes(bytes(TARGET_SAMPLE_RATE))

            with self.assertRaisesRegex(RecognitionError, "16-bit PCM"):
                _read_audio(Path(temporary.name))

    def test_masks_tokens_outside_vocabulary(self) -> None:
        logits = np.zeros((1, 2, 4), dtype=np.float32)
        masked = _mask_logits(logits, _Tokenizer(), ("a", "g"))
        self.assertTrue(np.isneginf(masked[..., 3]).all())
        self.assertFalse(np.isneginf(masked[..., :3]).any())

    def test_rejects_unsupported_phone(self) -> None:
        logits = np.zeros((1, 2, 4), dtype=np.float32)
        with self.assertRaisesRegex(UnsupportedPhoneError, "unsupported"):
            _mask_logits(logits, _Tokenizer(), ("x",))
