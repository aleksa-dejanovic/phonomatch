"""Record speech, transcribe it to IPA, and match it against known words."""

from .application import SoundAnalyzer, listen_and_match
from .domain import (
    DEFAULT_WORDS,
    AnalysisResult,
    CandidateScore,
    MatchResult,
    MatchSettings,
    decide_match,
    distance_ratio,
    normalize_ipa,
)
from .exceptions import (
    AudioRecordingError,
    RecognitionError,
    SoundAnalyzerError,
    UnsupportedPhoneError,
)
from .infrastructure.phonetics import phones_for_words
from .infrastructure.recognition import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_model,
)

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
    "load_model",
    "normalize_ipa",
    "phones_for_words",
]
