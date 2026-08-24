"""Record speech, transcribe it to IPA, and match it against known words."""

from .application.analyzer import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    PhonoMatch,
    listen_and_match,
)
from .domain import (
    DEFAULT_WORDS,
    AnalysisResult,
    CandidateScore,
    MatchResult,
    MatchSettings,
    PhraseAnalysisResult,
    WordAnalysisResult,
    decide_match,
    distance_ratio,
    normalize_ipa,
)
from .exceptions import (
    AudioRecordingError,
    OptionalDependencyError,
    PhonoMatchError,
    RecognitionError,
    UnsupportedPhoneError,
)
from .infrastructure.phonetics import phones_for_words


def load_model(*args: object, **kwargs: object) -> None:
    """Load the optional speech-recognition model."""
    from .application.analyzer import load_model as _load_model

    _load_model(*args, **kwargs)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_WORDS",
    "AnalysisResult",
    "AudioRecordingError",
    "CandidateScore",
    "MatchResult",
    "MatchSettings",
    "OptionalDependencyError",
    "PhonoMatch",
    "PhonoMatchError",
    "PhraseAnalysisResult",
    "RecognitionError",
    "UnsupportedPhoneError",
    "WordAnalysisResult",
    "decide_match",
    "distance_ratio",
    "listen_and_match",
    "load_model",
    "normalize_ipa",
    "phones_for_words",
]
