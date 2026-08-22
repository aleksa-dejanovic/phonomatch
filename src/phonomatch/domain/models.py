"""Result models returned by the analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Decision = Literal["likely_match", "ambiguous_or_unknown"]


@dataclass(frozen=True)
class CandidateScore:
    """The score assigned to one vocabulary entry."""

    word: str
    ipa: str
    raw_distance: float
    distance_ratio: float
    confidence: float


@dataclass(frozen=True)
class MatchResult:
    """The matching decision and its strongest candidates."""

    decision: Decision
    best_candidate: CandidateScore
    second_candidate: Optional[CandidateScore]
    relative_margin: float

    @property
    def accepted(self) -> bool:
        return self.decision == "likely_match"


@dataclass(frozen=True)
class AnalysisResult:
    """An IPA transcription together with its vocabulary match."""

    recognized_ipa: str
    match: MatchResult
    recognition_seconds: Optional[float] = None


@dataclass(frozen=True)
class WordAnalysisResult:
    """An independently matched word within a phrase."""

    recognized_ipa: str
    match: MatchResult
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PhraseAnalysisResult:
    """A sequence of independently matched vocabulary words."""

    words: tuple[WordAnalysisResult, ...]
    sequence_confidence: float
    sequence_accepted: bool
    alternative_words: Optional[tuple[str, ...]] = None
    recognition_seconds: Optional[float] = None

    @property
    def recognized_ipa(self) -> str:
        """Return the per-word transcriptions separated by spaces."""
        return " ".join(word.recognized_ipa for word in self.words)

    @property
    def accepted(self) -> bool:
        """Return whether every independently scored word was accepted."""
        return (
            self.sequence_accepted
            and bool(self.words)
            and all(word.match.accepted for word in self.words)
        )
