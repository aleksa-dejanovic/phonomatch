"""Wav2Vec2Phoneme integration with vocabulary-constrained CTC decoding."""

from __future__ import annotations

import math
import os
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from scipy.signal import resample_poly

from ..domain.ipa import normalize_ipa
from ..exceptions import RecognitionError, UnsupportedPhoneError
from .phonetics import phone_sequences_for_words
from .phrase_decoding import align_tokens, decode_phrases

DEFAULT_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class _ModelBundle:
    tokenizer: Any
    feature_extractor: Any
    model: Any


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


def speech_to_ipa(
    wav_path: Union[str, Path],
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
        inputs = bundle.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        import torch

        with torch.inference_mode():
            logits = bundle.model(**inputs).logits
        if allowed_phones is not None:
            logits = _mask_logits(logits, bundle.tokenizer, allowed_phones)

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = bundle.tokenizer.batch_decode(predicted_ids)[0]
    except (RecognitionError, UnsupportedPhoneError, ValueError):
        raise
    except Exception as exc:
        raise RecognitionError(f"speech recognition failed: {exc}") from exc
    return normalize_ipa(str(transcription))


def speech_to_phrase(
    wav_path: Union[str, Path],
    words: Mapping[str, str],
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
    if not words:
        raise ValueError("at least one vocabulary word is required")

    try:
        audio = _read_audio(path)
        bundle = _model_bundle(model_id, model_revision)
        logits = _model_logits(audio, bundle)
        vocabulary = bundle.tokenizer.get_vocab()
        phone_sequences = phone_sequences_for_words(words)
        unsupported = sorted(
            {
                phone
                for phones in phone_sequences.values()
                for phone in phones
                if phone not in vocabulary
            }
        )
        if unsupported:
            raise UnsupportedPhoneError(
                "phones unsupported by the Wav2Vec2Phoneme model: "
                + ", ".join(unsupported)
            )

        pronunciations = {
            word: tuple(vocabulary[phone] for phone in phones)
            for word, phones in phone_sequences.items()
        }
        blank_id = int(bundle.tokenizer.pad_token_id)
        allowed_ids = {
            token
            for pronunciation in pronunciations.values()
            for token in pronunciation
        }
        allowed_ids.add(blank_id)
        logits = _mask_token_ids(logits, allowed_ids)

        import torch

        log_probabilities = torch.log_softmax(logits[0], dim=-1).detach().cpu().numpy()
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
            predicted_ids = torch.argmax(
                logits[0, start_frame:end_frame], dim=-1
            ).unsqueeze(0)
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
    masked = logits.clone() if hasattr(logits, "clone") else logits.copy()
    masked[..., disallowed_ids] = float("-inf")
    return masked


def _model_logits(
    audio: np.ndarray[Any, np.dtype[np.float32]], bundle: _ModelBundle
) -> Any:
    inputs = bundle.feature_extractor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    import torch

    with torch.inference_mode():
        return bundle.model(**inputs).logits


def _hypothesis_confidence(hypotheses: tuple[Any, ...]) -> float:
    if len(hypotheses) == 1:
        return 1.0
    highest = max(float(hypothesis.score) for hypothesis in hypotheses)
    weights = [math.exp(float(hypothesis.score) - highest) for hypothesis in hypotheses]
    return weights[0] / sum(weights)


@lru_cache(maxsize=2)
def _model_bundle(model_id: str, revision: Optional[str]) -> _ModelBundle:
    # This checkpoint publishes PyTorch weights on its main revision. Without
    # this flag, Transformers also downloads an unmerged SafeTensors conversion
    # PR in the background, nearly doubling the Hugging Face cache footprint.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoModelForCTC,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
    )

    model_path = Path(model_id)
    is_local = model_path.is_dir()
    vocab_file = (
        str(model_path / "vocab.json")
        if is_local
        else hf_hub_download(model_id, filename="vocab.json", revision=revision)
    )
    tokenizer = Wav2Vec2CTCTokenizer(  # type: ignore[no-untyped-call]
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token=None,
    )
    if is_local or revision is None:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id, use_safetensors=False)
    else:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, revision=revision
        )
        model = AutoModelForCTC.from_pretrained(
            model_id,
            revision=revision,
            use_safetensors=False,
        )
    model.eval()
    return _ModelBundle(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model=model,
    )
