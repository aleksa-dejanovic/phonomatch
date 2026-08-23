import ctypes
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import numpy as np

from phonomatch import RecognitionError, UnsupportedPhoneError
from phonomatch.infrastructure.recognition import (
    TARGET_SAMPLE_RATE,
    _CTCTokenizer,
    _hypothesis_confidence,
    _log_softmax,
    _mask_logits,
    _mask_token_ids,
    _model_logits,
    _ModelBundle,
    _normalize_audio,
    _onnx_thread_count,
    _read_audio,
    _trim_allocator,
    load_model,
    speech_to_ipa,
    speech_to_phrase,
    unload_models,
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

    def test_ctc_tokenizer_reads_a_vocabulary_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as vocabulary:
            json.dump({"<pad>": 0, "<unk>": 1, "a": 2}, vocabulary)
            vocabulary.flush()
            tokenizer = _CTCTokenizer.from_file(vocabulary.name)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.batch_decode(np.array([[2, 2, 0, 2]])), ["aa"])

    def test_ctc_tokenizer_rejects_an_invalid_vocabulary(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as vocabulary:
            json.dump({"a": "not-an-id"}, vocabulary)
            vocabulary.flush()
            with self.assertRaisesRegex(ValueError, "<pad>"):
                _CTCTokenizer.from_file(vocabulary.name)

    def test_model_session_receives_normalized_batched_audio(self) -> None:
        tokenizer = _CTCTokenizer({"<pad>": 0}, 0, (0,))
        bundle = _ModelBundle(tokenizer, _Session())
        logits = _model_logits(np.array([1.0, 3.0], dtype=np.float32), bundle)
        self.assertEqual(logits.shape, (1, 2, 4))

    def test_log_softmax_normalizes_each_frame(self) -> None:
        result = _log_softmax(np.array([[1.0, 2.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.exp(result).sum()), 1.0)

    def test_normalize_audio_returns_zero_mean_unit_variance(self) -> None:
        normalized = _normalize_audio(np.array([1.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.var()), 1.0, places=6)

    def test_masks_token_ids_without_mutating_original_logits(self) -> None:
        logits = np.zeros((1, 1, 3), dtype=np.float32)
        masked = _mask_token_ids(logits, {0, 2})
        self.assertTrue(np.isneginf(masked[..., 1]).all())
        self.assertFalse(np.isneginf(logits).any())

    def test_hypothesis_confidence_handles_one_and_two_hypotheses(self) -> None:
        first = SimpleNamespace(score=10.0)
        second = SimpleNamespace(score=8.0)
        self.assertEqual(_hypothesis_confidence((first,)), 1.0)
        self.assertGreater(_hypothesis_confidence((first, second)), 0.5)

    def test_load_model_wraps_loader_errors(self) -> None:
        with (
            patch(
                "phonomatch.infrastructure.recognition._model_bundle",
                side_effect=RuntimeError("missing model"),
            ),
            self.assertRaisesRegex(RecognitionError, "missing model"),
        ):
            load_model()

    def test_unload_models_clears_the_model_cache(self) -> None:
        with (
            patch(
                "phonomatch.infrastructure.recognition._model_bundle.cache_clear"
            ) as clear_cache,
            patch(
                "phonomatch.infrastructure.recognition._trim_allocator"
            ) as trim_allocator,
        ):
            unload_models()

        clear_cache.assert_called_once_with()
        trim_allocator.assert_called_once_with()

    def test_trim_allocator_uses_malloc_trim_on_linux(self) -> None:
        library = SimpleNamespace(malloc_trim=Mock())
        with (
            patch("phonomatch.infrastructure.recognition.sys.platform", "linux"),
            patch(
                "phonomatch.infrastructure.recognition.ctypes.CDLL",
                return_value=library,
            ),
        ):
            _trim_allocator()

        self.assertEqual(library.malloc_trim.argtypes, [ctypes.c_size_t])
        self.assertIs(library.malloc_trim.restype, ctypes.c_int)
        self.assertEqual(library.malloc_trim.call_args.args, (0,))

    def test_trim_allocator_is_skipped_off_linux(self) -> None:
        with (
            patch("phonomatch.infrastructure.recognition.sys.platform", "darwin"),
            patch("phonomatch.infrastructure.recognition.ctypes.CDLL") as load_library,
        ):
            _trim_allocator()

        load_library.assert_not_called()

    def test_speech_to_ipa_decodes_mocked_model_logits(self) -> None:
        tokenizer = _CTCTokenizer({"<pad>": 0, "a": 1}, 0, (0,))
        bundle = _ModelBundle(tokenizer, object())
        logits = np.array([[[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
        with (
            patch(
                "phonomatch.infrastructure.recognition.Path.is_file", return_value=True
            ),
            patch(
                "phonomatch.infrastructure.recognition._read_audio",
                return_value=np.zeros(2, dtype=np.float32),
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_bundle",
                return_value=bundle,
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_logits",
                return_value=logits,
            ),
        ):
            result = speech_to_ipa("recording.wav", allowed_phones=("a",))
        self.assertEqual(result, "aa")

    def test_speech_recognition_rejects_missing_or_non_wav_paths(self) -> None:
        with self.assertRaisesRegex(RecognitionError, "does not exist"):
            speech_to_ipa("missing.wav")
        with (
            tempfile.NamedTemporaryFile(suffix=".mp3") as audio,
            self.assertRaisesRegex(RecognitionError, "requires WAV"),
        ):
            speech_to_ipa(audio.name)

    def test_phrase_decoding_uses_mocked_ctc_hypothesis(self) -> None:
        tokenizer = _CTCTokenizer({"<pad>": 0, "a": 1}, 0, (0,))
        bundle = _ModelBundle(tokenizer, object())
        hypothesis = SimpleNamespace(words=("word",), token_ids=(1,), score=3.0)
        logits = np.array([[[0.0, 2.0], [0.0, 2.0]]])
        span = SimpleNamespace(start=0, end=2)
        with (
            patch(
                "phonomatch.infrastructure.recognition.Path.is_file", return_value=True
            ),
            patch(
                "phonomatch.infrastructure.recognition._read_audio",
                return_value=np.zeros(16_000, dtype=np.float32),
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_bundle",
                return_value=bundle,
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_logits",
                return_value=logits,
            ),
            patch(
                "phonomatch.infrastructure.recognition.phone_sequences_for_words",
                return_value={"word": ("a",)},
            ),
            patch(
                "phonomatch.infrastructure.recognition.decode_phrases",
                return_value=(hypothesis,),
            ),
            patch(
                "phonomatch.infrastructure.recognition.align_tokens",
                return_value=(span,),
            ),
        ):
            result = speech_to_phrase("recording.wav", {"word": "a"})
        self.assertEqual(result.words[0].word, "word")
        self.assertEqual(result.words[0].ipa, "a")
        self.assertEqual(result.confidence, 1.0)

    def test_phrase_decoding_rejects_empty_or_unsupported_vocabulary(self) -> None:
        with (
            patch(
                "phonomatch.infrastructure.recognition.Path.is_file", return_value=True
            ),
            self.assertRaisesRegex(ValueError, "at least one vocabulary"),
        ):
            speech_to_phrase("recording.wav", {})
        tokenizer = _CTCTokenizer({"<pad>": 0, "a": 1}, 0, (0,))
        bundle = _ModelBundle(tokenizer, object())
        with (
            patch(
                "phonomatch.infrastructure.recognition.Path.is_file", return_value=True
            ),
            patch(
                "phonomatch.infrastructure.recognition._read_audio",
                return_value=np.zeros(1, dtype=np.float32),
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_bundle",
                return_value=bundle,
            ),
            patch(
                "phonomatch.infrastructure.recognition._model_logits",
                return_value=np.zeros((1, 1, 2)),
            ),
            patch(
                "phonomatch.infrastructure.recognition.phone_sequences_for_words",
                return_value={"word": ("x",)},
            ),
            self.assertRaisesRegex(UnsupportedPhoneError, "x"),
        ):
            speech_to_phrase("recording.wav", {"word": "x"})

    def test_onnx_threads_default_to_eight_or_cpu_count(self) -> None:
        with patch(
            "phonomatch.infrastructure.recognition.os.cpu_count", return_value=16
        ):
            self.assertEqual(_onnx_thread_count(), 8)

    def test_onnx_threads_accepts_a_positive_environment_override(self) -> None:
        with patch.dict("os.environ", {"PHONOMATCH_ONNX_THREADS": "4"}):
            self.assertEqual(_onnx_thread_count(), 4)

    def test_onnx_threads_rejects_an_invalid_environment_override(self) -> None:
        with (
            patch.dict("os.environ", {"PHONOMATCH_ONNX_THREADS": "zero"}),
            self.assertRaisesRegex(ValueError, "positive integer"),
        ):
            _onnx_thread_count()
