"""ONNX Wav2Vec2Phoneme integration with vocabulary-constrained CTC decoding."""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from gc import collect
from pathlib import Path
from typing import Any, Optional, cast

from ..domain.ipa import normalize_ipa
from ..exceptions import (
    OptionalDependencyError,
    RecognitionError,
    UnsupportedPhoneError,
)
from .phonetics import Vocabulary

try:
    import numpy as np
    from scipy.signal import resample_poly
except ModuleNotFoundError as exc:
    if exc.name in {"numpy", "scipy"}:
        raise OptionalDependencyError("recognition") from exc
    raise

from .phrase_decoding import align_tokens, decode_phrases

DEFAULT_MODEL_ID = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
DEFAULT_MODEL_REVISION = "c69750f5043e5e1f8a71ab95dd3b98338c280c92"
DEFAULT_MODEL_FILE = "onnx/model_q4f16.onnx"
DEFAULT_ONNX_THREADS = 8
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class _ModelBundle:
    tokenizer: _CTCTokenizer
    session: Any


@dataclass(frozen=True)
class _CTCTokenizer:
    """Minimal CTC decoder for the model's JSON vocabulary.

    Keeping this adapter local avoids importing Transformers (and therefore a
    framework backend) merely to turn predicted token IDs into IPA text.
    """

    vocabulary: Mapping[str, int]
    pad_token_id: int
    all_special_ids: tuple[int, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> _CTCTokenizer:
        vocabulary = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(vocabulary, dict) or "<pad>" not in vocabulary:
            raise ValueError("model vocabulary is missing its <pad> token")
        if not all(
            isinstance(token, str) and isinstance(index, int)
            for token, index in vocabulary.items()
        ):
            raise ValueError("model vocabulary must map token strings to IDs")
        special_ids = tuple(
            index
            for token, index in vocabulary.items()
            if token.startswith("<") and token.endswith(">")
        )
        return cls(vocabulary, vocabulary["<pad>"], special_ids)

    def get_vocab(self) -> Mapping[str, int]:
        return self.vocabulary

    def batch_decode(self, token_batches: np.ndarray[Any, Any]) -> list[str]:
        token_by_id = {index: token for token, index in self.vocabulary.items()}
        decoded: list[str] = []
        for token_ids in token_batches:
            previous_id: Optional[int] = None
            tokens: list[str] = []
            for token_id in token_ids:
                index = int(token_id)
                if index != previous_id and index != self.pad_token_id:
                    token = token_by_id.get(index)
                    if token is None:
                        raise ValueError(f"model produced unknown token ID: {index}")
                    if index not in self.all_special_ids:
                        tokens.append(token)
                previous_id = index
            decoded.append("".join(tokens))
        return decoded


@dataclass(frozen=True)
class RecognizedWord:
    """One decoded word and its approximate location in the recording."""

    word: str
    ipa: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PhraseTranscription:
    """A lexicon-constrained phrase transcription."""

    words: tuple[RecognizedWord, ...]
    confidence: float
    alternative: Optional[tuple[str, ...]]


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
) -> None:
    """Load and cache the recognition model without processing audio."""
    try:
        _model_bundle(model_id, model_revision)
    except Exception as exc:
        raise RecognitionError(
            f"could not load speech recognition model: {exc}"
        ) from exc


def unload_models() -> None:
    """Release cached recognition sessions so their memory can be reclaimed.

    This only releases in-process model objects. Downloaded model files remain in
    the Hugging Face cache, so a later :func:`load_model` normally needs no
    network access. On Linux, also ask the C allocator to return unused heap
    pages to the operating system when the platform supports it.
    """
    _model_bundle.cache_clear()
    collect()
    _trim_allocator()


def _trim_allocator() -> None:
    """Return unused glibc heap pages to Linux when ``malloc_trim`` is present."""
    if platform.system() != "Linux":
        return
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def speech_to_ipa(
    wav_path: str | Path,
    allowed_phones: Optional[Iterable[str]] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
) -> str:
    """Transcribe a PCM WAV file to IPA with Wav2Vec2Phoneme.

    When ``allowed_phones`` is supplied, tokens outside that inventory are
    masked before CTC decoding. The model is downloaded once by Hugging Face
    and then reused from its local cache.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise RecognitionError(f"audio file does not exist: {path}")
    if path.suffix.lower() != ".wav":
        raise RecognitionError("Wav2Vec2Phoneme requires WAV input")

    try:
        audio = _read_audio(path)
        bundle = _model_bundle(model_id, model_revision)
        logits = _model_logits(audio, bundle)
        if allowed_phones is not None:
            logits = _mask_logits(logits, bundle.tokenizer, allowed_phones)

        predicted_ids = np.argmax(logits, axis=-1)
        transcription = bundle.tokenizer.batch_decode(predicted_ids)[0]
    except (RecognitionError, UnsupportedPhoneError, ValueError):
        raise
    except Exception as exc:
        raise RecognitionError(f"speech recognition failed: {exc}") from exc
    return normalize_ipa(str(transcription))


def speech_to_phrase(
    wav_path: str | Path,
    vocabulary: Vocabulary,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
    beam_size: int = 64,
    max_words: int = 12,
) -> PhraseTranscription:
    """Decode a phrase as an arbitrary sequence of vocabulary words."""
    path = Path(wav_path)
    if not path.is_file():
        raise RecognitionError(f"audio file does not exist: {path}")
    if path.suffix.lower() != ".wav":
        raise RecognitionError("Wav2Vec2Phoneme requires WAV input")
    try:
        audio = _read_audio(path)
        bundle = _model_bundle(model_id, model_revision)
        logits = _model_logits(audio, bundle)
        token_vocabulary = bundle.tokenizer.get_vocab()
        word_phone_sequences = vocabulary.phone_sequences
        unsupported = sorted(
            {
                phone
                for phones in word_phone_sequences.values()
                for phone in phones
                if phone not in token_vocabulary
            }
        )
        if unsupported:
            raise UnsupportedPhoneError(
                "phones unsupported by the Wav2Vec2Phoneme model: "
                + ", ".join(unsupported)
            )

        pronunciations = {
            word: tuple(token_vocabulary[phone] for phone in phones)
            for word, phones in word_phone_sequences.items()
        }
        blank_id = int(bundle.tokenizer.pad_token_id)
        allowed_ids = {
            token
            for pronunciation in pronunciations.values()
            for token in pronunciation
        }
        allowed_ids.add(blank_id)
        logits = _mask_token_ids(logits, allowed_ids)

        log_probabilities = _log_softmax(logits[0])
        hypotheses = decode_phrases(
            log_probabilities,
            pronunciations,
            blank_id,
            beam_size=beam_size,
            nbest=2,
            max_words=max_words,
        )
        if not hypotheses:
            raise RecognitionError("no vocabulary phrase could be decoded")

        best = hypotheses[0]
        spans = align_tokens(log_probabilities, best.token_ids, blank_id)
        duration = len(audio) / TARGET_SAMPLE_RATE
        frame_count = log_probabilities.shape[0]
        recognized_words: list[RecognizedWord] = []
        token_offset = 0
        for word in best.words:
            token_length = len(pronunciations[word])
            word_spans = spans[token_offset : token_offset + token_length]
            start_frame = word_spans[0].start
            end_frame = word_spans[-1].end
            predicted_ids = np.argmax(logits[0, start_frame:end_frame], axis=-1)[
                np.newaxis, :
            ]
            recognized_ipa = normalize_ipa(
                str(bundle.tokenizer.batch_decode(predicted_ids)[0])
            )
            recognized_words.append(
                RecognizedWord(
                    word=word,
                    ipa=recognized_ipa,
                    start_seconds=start_frame / frame_count * duration,
                    end_seconds=end_frame / frame_count * duration,
                )
            )
            token_offset += token_length

        confidence = _hypothesis_confidence(hypotheses)
        alternative = hypotheses[1].words if len(hypotheses) > 1 else None
        return PhraseTranscription(tuple(recognized_words), confidence, alternative)
    except (RecognitionError, UnsupportedPhoneError, ValueError):
        raise
    except Exception as exc:
        raise RecognitionError(f"phrase recognition failed: {exc}") from exc


def _read_audio(path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise RecognitionError(f"could not read WAV file: {exc}") from exc

    if channels < 1 or sample_rate < 1 or frame_count < 1:
        raise RecognitionError("WAV file contains no usable audio")
    if sample_width != 2:
        raise RecognitionError("WAV input must contain 16-bit PCM samples")

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio /= float(np.iinfo(np.int16).max)

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    return audio


def _mask_logits(logits: Any, tokenizer: Any, phones: Iterable[str]) -> Any:
    normalized_phones = tuple(dict.fromkeys(normalize_ipa(phone) for phone in phones))
    if not normalized_phones:
        raise ValueError("allowed_phones must contain at least one phone")

    vocabulary = tokenizer.get_vocab()
    unsupported = [phone for phone in normalized_phones if phone not in vocabulary]
    if unsupported:
        raise UnsupportedPhoneError(
            "phones unsupported by the Wav2Vec2Phoneme model: " + ", ".join(unsupported)
        )

    allowed_ids = {vocabulary[phone] for phone in normalized_phones}
    allowed_ids.update(tokenizer.all_special_ids)
    disallowed_ids = sorted(set(range(logits.shape[-1])) - allowed_ids)
    if disallowed_ids:
        # Model outputs created in inference mode cannot be changed in place
        # after leaving that context; cloning also preserves the caller's data.
        logits = logits.clone() if hasattr(logits, "clone") else logits.copy()
        logits[..., disallowed_ids] = float("-inf")
    return logits


def _mask_token_ids(logits: Any, allowed_ids: set[int]) -> Any:
    disallowed_ids = sorted(set(range(logits.shape[-1])) - allowed_ids)
    if not disallowed_ids:
        return logits
    masked = logits.copy()
    masked[..., disallowed_ids] = float("-inf")
    return masked


def _model_logits(
    audio: np.ndarray[Any, np.dtype[np.float32]], bundle: _ModelBundle
) -> np.ndarray[Any, np.dtype[np.float32]]:
    normalized = _normalize_audio(audio)[np.newaxis, :]
    model_input = bundle.session.get_inputs()[0].name
    return cast(
        np.ndarray[Any, np.dtype[np.float32]],
        np.asarray(
            bundle.session.run(None, {model_input: normalized})[0], dtype=np.float32
        ),
    )


def _normalize_audio(
    audio: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Match Wav2Vec2FeatureExtractor's unpadded zero-mean normalization."""
    mean = float(np.mean(audio))
    variance = float(np.var(audio))
    return ((audio - mean) / math.sqrt(variance + 1e-7)).astype(np.float32)


def _log_softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    maximum = np.max(logits, axis=-1, keepdims=True)
    return cast(
        np.ndarray[Any, np.dtype[np.float64]],
        (
            logits
            - maximum
            - np.log(np.sum(np.exp(logits - maximum), axis=-1, keepdims=True))
        ).astype(np.float64),
    )


def _onnx_thread_count() -> int:
    """Return the configured inference-thread limit for one recognition session."""
    configured = os.environ.get("PHONOMATCH_ONNX_THREADS")
    if configured is None:
        return min(DEFAULT_ONNX_THREADS, os.cpu_count() or 1)
    try:
        thread_count = int(configured)
    except ValueError as exc:
        raise ValueError("PHONOMATCH_ONNX_THREADS must be a positive integer") from exc
    if thread_count < 1:
        raise ValueError("PHONOMATCH_ONNX_THREADS must be a positive integer")
    return thread_count


def _hypothesis_confidence(hypotheses: tuple[Any, ...]) -> float:
    if len(hypotheses) == 1:
        return 1.0
    highest = max(float(hypothesis.score) for hypothesis in hypotheses)
    weights = [math.exp(float(hypothesis.score) - highest) for hypothesis in hypotheses]
    return weights[0] / sum(weights)


@lru_cache(maxsize=2)
def _model_bundle(model_id: str, revision: Optional[str]) -> _ModelBundle:
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        if exc.name in {"onnxruntime", "huggingface_hub"}:
            raise OptionalDependencyError("recognition") from exc
        raise

    model_path = Path(model_id)
    is_local = model_path.is_dir()
    vocab_file = (
        str(model_path / "vocab.json")
        if is_local
        else hf_hub_download(model_id, filename="vocab.json", revision=revision)
    )
    onnx_file = (
        str(model_path / DEFAULT_MODEL_FILE)
        if is_local
        else hf_hub_download(model_id, filename=DEFAULT_MODEL_FILE, revision=revision)
    )
    session_options = onnxruntime.SessionOptions()
    # The published q4f16 graph is valid, but ORT's SimplifiedLayerNorm fusion
    # fails to initialize it on the last release supporting Python 3.10.
    # Disabling graph fusions keeps the model portable across every supported
    # Python version; the quantized graph remains substantially smaller than
    # the former PyTorch checkpoint.
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session_options.intra_op_num_threads = _onnx_thread_count()
    session_options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(onnx_file, sess_options=session_options)
    return _ModelBundle(_CTCTokenizer.from_file(vocab_file), session)
