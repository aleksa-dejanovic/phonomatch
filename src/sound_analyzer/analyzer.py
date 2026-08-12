"""High-level orchestration API for sound analysis."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Union

from .audio import recorded_audio
from .config import DEFAULT_SETTINGS, DEFAULT_WORDS, MatchSettings
from .matching import decide_match
from .models import AnalysisResult
from .phonetics import (
    phones_for_words,
    phonetic_distance,
    phonetic_maximum_distance,
)
from .recognition import speech_to_ipa


class SoundAnalyzer:
    """Transcribe recordings and match them against an IPA vocabulary."""

    def __init__(
        self,
        words: Mapping[str, str] = DEFAULT_WORDS,
        settings: MatchSettings = DEFAULT_SETTINGS,
    ) -> None:
        if not words:
            raise ValueError("at least one vocabulary word is required")
        self._words = dict(words)
        self._settings = settings
        self._phones = phones_for_words(self._words)

    @property
    def words(self) -> Mapping[str, str]:
        """Return a defensive copy of the analyzer vocabulary."""
        return dict(self._words)

    def analyze_file(self, wav_path: Union[str, Path]) -> AnalysisResult:
        """Transcribe and match an existing WAV file."""
        ipa = speech_to_ipa(wav_path, self._phones)
        return self.match_ipa(ipa)

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

    def listen(self, *, seconds: float = 2.0) -> AnalysisResult:
        """Record from the default microphone, transcribe, and match."""
        with recorded_audio(seconds) as wav_path:
            return self.analyze_file(wav_path)


def listen_and_match(
    words: Mapping[str, str] = DEFAULT_WORDS,
    *,
    seconds: float = 2.0,
    settings: MatchSettings = DEFAULT_SETTINGS,
) -> AnalysisResult:
    """Convenience wrapper around :class:`SoundAnalyzer`."""
    return SoundAnalyzer(words, settings).listen(seconds=seconds)
