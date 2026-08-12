"""Record speech, transcribe it to IPA, and match it against known words."""

from .analyzer import (
    DEFAULT_WORDS,
    MatchResult,
    decide_match,
    listen_and_match,
    normalize_ipa,
)
from .cli import main

__all__ = [
    "DEFAULT_WORDS",
    "MatchResult",
    "decide_match",
    "listen_and_match",
    "main",
    "normalize_ipa",
]
