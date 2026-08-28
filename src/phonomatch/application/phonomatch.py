"""The public façade for recording, recognition, and pronunciation matching."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional

from ..domain.config import DEFAULT_SETTINGS, MatchSettings
from ..domain.models import AnalysisResult, PhraseAnalysisResult
from ..infrastructure.audio import recorded_audio
from .analyzer import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Analyzer

Recording = Callable[[float], AbstractContextManager[Path]]


class PhonoMatch:
    """Coordinate microphone capture with pronunciation analysis."""

    def __init__(
        self,
        words: Mapping[str, str] | None = None,
        settings: MatchSettings = DEFAULT_SETTINGS,
        *,
        default_words: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
        record: Recording = recorded_audio,
    ) -> None:
        self._analyzer = Analyzer(
            words,
            settings,
            default_words=default_words,
            model_id=model_id,
            model_revision=model_revision,
        )
        self._record = record

    @property
    def words(self) -> Mapping[str, str]:
        """Return a defensive copy of the configured vocabulary."""
        return self._analyzer.words

    def load_model(self) -> None:
        """Load the speech model now so later recognition avoids startup delay."""
        self._analyzer.load_model()

    def unload_model(self) -> None:
        """Release the cached speech model."""
        self._analyzer.unload_model()

    def match_ipa(self, ipa: str) -> AnalysisResult:
        """Match an existing IPA transcription."""
        return self._analyzer.match_ipa(ipa)

    def analyze_file(self, wav_path: str | Path) -> AnalysisResult:
        """Analyze an existing WAV file."""
        return self._analyzer.analyze_file(wav_path)

    def analyze_phrase_file(
        self,
        wav_path: str | Path,
        *,
        beam_size: int = 64,
        max_words: int = 12,
    ) -> PhraseAnalysisResult:
        """Analyze an existing WAV file as a vocabulary word sequence."""
        return self._analyzer.analyze_phrase_file(
            wav_path, beam_size=beam_size, max_words=max_words
        )

    def record_and_analyze_word(
        self,
        *,
        seconds: float = 2.0,
        on_recording_complete: Optional[Callable[[], None]] = None,
    ) -> AnalysisResult:
        """Capture one utterance, then analyze its temporary WAV file."""
        self.load_model()
        with self._record(seconds) as wav_path:
            if on_recording_complete is not None:
                on_recording_complete()
            return self.analyze_file(wav_path)

    def record_and_analyze_phrase(
        self,
        *,
        seconds: float = 4.0,
        beam_size: int = 64,
        max_words: int = 12,
        on_recording_complete: Optional[Callable[[], None]] = None,
    ) -> PhraseAnalysisResult:
        """Capture an utterance, then decode it as a vocabulary word sequence."""
        self.load_model()
        with self._record(seconds) as wav_path:
            if on_recording_complete is not None:
                on_recording_complete()
            return self.analyze_phrase_file(
                wav_path,
                beam_size=beam_size,
                max_words=max_words,
            )


def listen_and_match(
    words: Mapping[str, str] | None = None,
    *,
    seconds: float = 2.0,
    settings: MatchSettings = DEFAULT_SETTINGS,
    default_words: bool = False,
) -> AnalysisResult:
    """Backward-compatible convenience workflow for one live utterance."""
    return PhonoMatch(
        words, settings, default_words=default_words
    ).record_and_analyze_word(seconds=seconds)
