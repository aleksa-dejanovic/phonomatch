"""Record speech, transcribe it to IPA, and match it against known words."""

from .analyzer import SoundAnalyzer, listen_and_match
from .config import DEFAULT_WORDS, MatchSettings
from .exceptions import (
    AudioRecordingError,
    RecognitionError,
    SoundAnalyzerError,
    UnsupportedPhoneError,
)
from .ipa import normalize_ipa
from .matching import decide_match, distance_ratio
from .models import AnalysisResult, CandidateScore, MatchResult
from .phonetics import phones_for_words
from .recognition import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_WORDS",
    "AnalysisResult",
    "AudioRecordingError",
    "CandidateScore",
    "MatchResult",
    "MatchSettings",
    "RecognitionError",
    "SoundAnalyzer",
    "SoundAnalyzerError",
    "UnsupportedPhoneError",
    "decide_match",
    "distance_ratio",
    "listen_and_match",
    "normalize_ipa",
    "phones_for_words",
]
