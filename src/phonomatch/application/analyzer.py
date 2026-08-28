"""High-level orchestration API for sound analysis."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ..domain.config import DEFAULT_SETTINGS, MatchSettings
from ..domain.matching import decide_match
from ..domain.models import AnalysisResult, PhraseAnalysisResult, WordAnalysisResult
from ..exceptions import OptionalDependencyError
from ..infrastructure.phonetics import (
    Vocabulary,
    phonetic_distance,
    phonetic_maximum_distance,
)

DEFAULT_MODEL_ID = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
DEFAULT_MODEL_REVISION = "c69750f5043e5e1f8a71ab95dd3b98338c280c92"


def _recognition_module() -> Any:
    """Load speech recognition only when a recognition API is used."""
    try:
        from ..infrastructure import recognition
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "scipy", "onnxruntime", "huggingface_hub"}:
            raise OptionalDependencyError("recognition") from exc
        raise
    return recognition


def speech_to_ipa(*args: object, **kwargs: object) -> str:
    """Lazily dispatch to the optional speech-recognition integration."""
    return str(_recognition_module().speech_to_ipa(*args, **kwargs))


def speech_to_phrase(*args: object, **kwargs: object) -> Any:
    """Lazily dispatch to the optional phrase-recognition integration."""
    return _recognition_module().speech_to_phrase(*args, **kwargs)


def load_model(*args: object, **kwargs: object) -> None:
    """Lazily load the optional speech-recognition model."""
    _recognition_module().load_model(*args, **kwargs)


def unload_models() -> None:
    """Lazily release optional cached speech-recognition models."""
    _recognition_module().unload_models()


class Analyzer:
    """Transcribe recordings and match them against an IPA vocabulary."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        settings: MatchSettings = DEFAULT_SETTINGS,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ) -> None:
        self._vocabulary = vocabulary
        self._settings = settings
        self._model_id = model_id
        self._model_revision = model_revision

    @property
    def words(self) -> Mapping[str, str]:
        """Return a defensive copy of the analyzer vocabulary."""
        return dict(self._vocabulary.words)

    def analyze_file(self, wav_path: str | Path) -> AnalysisResult:
        """Transcribe and match an existing WAV file."""
        started_at = time.perf_counter()
        ipa = speech_to_ipa(
            wav_path,
            self._vocabulary.phones,
            model_id=self._model_id,
            model_revision=self._model_revision,
        )
        result = self.match_ipa(ipa)
        return AnalysisResult(
            recognized_ipa=result.recognized_ipa,
            match=result.match,
            recognition_seconds=time.perf_counter() - started_at,
        )

    def analyze_phrase_file(
        self,
        wav_path: str | Path,
        *,
        beam_size: int = 64,
        max_words: int = 12,
    ) -> PhraseAnalysisResult:
        """Decode and independently match vocabulary words in a WAV file."""
        started_at = time.perf_counter()
        transcription = speech_to_phrase(
            wav_path,
            self._vocabulary,
            model_id=self._model_id,
            model_revision=self._model_revision,
            beam_size=beam_size,
            max_words=max_words,
        )
        word_results = tuple(
            WordAnalysisResult(
                recognized_ipa=word.ipa,
                match=self.match_ipa(word.ipa).match,
                start_seconds=word.start_seconds,
                end_seconds=word.end_seconds,
            )
            for word in transcription.words
        )
        return PhraseAnalysisResult(
            words=word_results,
            sequence_confidence=transcription.confidence,
            sequence_accepted=(
                transcription.confidence >= self._settings.min_confidence
            ),
            alternative_words=transcription.alternative,
            recognition_seconds=time.perf_counter() - started_at,
        )

    def load_model(self) -> None:
        """Load the speech model now so later recognition avoids startup delay."""
        load_model(self._model_id, self._model_revision)

    def unload_model(self) -> None:
        """Release the cached speech model and request Python memory cleanup."""
        unload_models()

    def match_ipa(self, ipa: str) -> AnalysisResult:
        """Match an existing IPA transcription without recording audio."""
        match = decide_match(
            ipa,
            self._vocabulary.words,
            phonetic_distance,
            maximum_distance_function=phonetic_maximum_distance,
            settings=self._settings,
        )
        return AnalysisResult(recognized_ipa=ipa, match=match)
