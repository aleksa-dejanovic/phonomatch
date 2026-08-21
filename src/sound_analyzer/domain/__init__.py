"""Pure domain models, configuration, normalization, and matching rules."""

from .config import DEFAULT_SETTINGS, DEFAULT_WORDS, MatchSettings
from .ipa import normalize_ipa
from .matching import decide_match, distance_ratio
from .models import (
    AnalysisResult,
    CandidateScore,
    MatchResult,
    PhraseAnalysisResult,
    WordAnalysisResult,
)

__all__ = [
    "DEFAULT_SETTINGS",
    "DEFAULT_WORDS",
    "AnalysisResult",
    "CandidateScore",
    "MatchResult",
    "MatchSettings",
    "PhraseAnalysisResult",
    "WordAnalysisResult",
    "decide_match",
    "distance_ratio",
    "normalize_ipa",
]
