"""Public result models returned by the analyzer."""

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
