import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from phonomatch import RecognitionError, UnsupportedPhoneError
from phonomatch.infrastructure.recognition import (
    TARGET_SAMPLE_RATE,
    _CTCTokenizer,
    _log_softmax,
    _mask_logits,
    _model_logits,
    _ModelBundle,
    _read_audio,
)


class _Tokenizer:
    all_special_ids: ClassVar[list[int]] = [0]

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<pad>": 0, "a": 1, "ɡ": 2, "ʃ": 3}


class _Session:
    class _Input:
        name = "input_values"

    def get_inputs(self) -> list["_Session._Input"]:
        return [self._Input()]

    @staticmethod
    def run(
        _outputs: object, inputs: dict[str, np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        input_values = inputs["input_values"]
        return [np.zeros((1, input_values.shape[1], 4), dtype=np.float32)]


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

    def test_ctc_decoder_collapses_repeated_tokens_and_blanks(self) -> None:
        tokenizer = _CTCTokenizer({"<pad>": 0, "a": 1, "ɡ": 2, "<unk>": 3}, 0, (0, 3))
        decoded = tokenizer.batch_decode(np.array([[1, 1, 0, 1, 2, 2, 3]]))
        self.assertEqual(decoded, ["aaɡ"])

    def test_model_session_receives_normalized_batched_audio(self) -> None:
        tokenizer = _CTCTokenizer({"<pad>": 0}, 0, (0,))
        bundle = _ModelBundle(tokenizer, _Session())
        logits = _model_logits(np.array([1.0, 3.0], dtype=np.float32), bundle)
        self.assertEqual(logits.shape, (1, 2, 4))

    def test_log_softmax_normalizes_each_frame(self) -> None:
        result = _log_softmax(np.array([[1.0, 2.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.exp(result).sum()), 1.0)
