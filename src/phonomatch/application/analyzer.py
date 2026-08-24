"""High-level orchestration API for sound analysis."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from ..domain.config import DEFAULT_SETTINGS, DEFAULT_WORDS, MatchSettings
from ..domain.matching import decide_match
from ..domain.models import AnalysisResult, PhraseAnalysisResult, WordAnalysisResult
from ..exceptions import OptionalDependencyError
from ..infrastructure.audio import recorded_audio
from ..infrastructure.phonetics import (
    phones_for_words,
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


class PhonoMatch:
    """Transcribe recordings and match them against an IPA vocabulary."""

    def __init__(
        self,
        words: Mapping[str, str] = DEFAULT_WORDS,
        settings: MatchSettings = DEFAULT_SETTINGS,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ) -> None:
        if not words:
            raise ValueError("at least one vocabulary word is required")
        self._words = dict(words)
        self._settings = settings
        self._phones = phones_for_words(self._words)
        self._model_id = model_id
        self._model_revision = model_revision

    @property
    def words(self) -> Mapping[str, str]:
        """Return a defensive copy of the analyzer vocabulary."""
        return dict(self._words)

    def analyze_file(self, wav_path: str | Path) -> AnalysisResult:
        """Transcribe and match an existing WAV file."""
        started_at = time.perf_counter()
        ipa = speech_to_ipa(
            wav_path,
            self._phones,
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
            self._words,
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
            self._words,
            phonetic_distance,
            maximum_distance_function=phonetic_maximum_distance,
            settings=self._settings,
        )
        return AnalysisResult(recognized_ipa=ipa, match=match)

    def listen(
        self,
        *,
        seconds: float = 2.0,
        on_recording_complete: Optional[Callable[[], None]] = None,
    ) -> AnalysisResult:
        """Record from the default microphone, transcribe, and match."""
        self.load_model()
        with recorded_audio(seconds) as wav_path:
            if on_recording_complete is not None:
                on_recording_complete()
            return self.analyze_file(wav_path)

    def listen_for_phrase(
        self,
        *,
        seconds: float = 4.0,
        beam_size: int = 64,
        max_words: int = 12,
        on_recording_complete: Optional[Callable[[], None]] = None,
    ) -> PhraseAnalysisResult:
        """Record and decode an arbitrary sequence of vocabulary words."""
        self.load_model()
        with recorded_audio(seconds) as wav_path:
            if on_recording_complete is not None:
                on_recording_complete()
            return self.analyze_phrase_file(
                wav_path,
                beam_size=beam_size,
                max_words=max_words,
            )


def listen_and_match(
    words: Mapping[str, str] = DEFAULT_WORDS,
    *,
    seconds: float = 2.0,
    settings: MatchSettings = DEFAULT_SETTINGS,
) -> AnalysisResult:
    """Convenience wrapper around :class:`PhonoMatch`."""
    return PhonoMatch(words, settings).listen(seconds=seconds)
