"""Core sound recording and phonetic matching functionality."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import tempfile
from typing import Optional, Union
import unicodedata
import wave


DEFAULT_WORDS: dict[str, str] = {
    "naku": "naku",
    "selim": "selim",
    "tova": "tova",
    "grun": "ɡrun",
    "shaki": "ʃaki",
    "flabkiver": "flæbˈkɪvər",
    "dracarys": "draˈkarys",
    "lykiri": "lyˈkiri",
    "dohaeras": "dohaeˈra:s",
}

# Transcribers, keyboards, and fonts often emit different code points for the
# same IPA notation. PanPhon does not interpret all of these aliases equally,
# so canonicalize only notation variants that do not change the sound.
IPA_EQUIVALENTS = str.maketrans({
    "g": "ɡ",   # Latin g -> IPA script g
    ":": "ː",   # ASCII colon -> IPA length mark
    "꞉": "ː",   # modifier-letter colon -> IPA length mark
    "'": "ʼ",   # apostrophe variants -> IPA modifier apostrophe
    "‘": "ʼ",
    "’": "ʼ",
    "ʹ": "ʼ",
    "′": "ʼ",
    "͜": "͡",   # below tie bar -> canonical above tie bar
})


@dataclass(frozen=True)
class CandidateScore:
    word: str
    ipa: str
    distance: float
    confidence: float


@dataclass(frozen=True)
class MatchResult:
    decision: str
    best_candidate: CandidateScore
    second_candidate: Optional[CandidateScore]
    margin: float

    @property
    def accepted(self) -> bool:
        return self.decision == "likely_match"


def normalize_ipa(ipa: str) -> str:
    """Remove phone separators and canonicalize equivalent IPA symbols."""
    compact = "".join(ipa.split()).translate(IPA_EQUIVALENTS)
    return unicodedata.normalize("NFC", compact)


def normalized_distance(cost: float, ipa_a: str, ipa_b: str) -> float:
    return cost / max(len(normalize_ipa(ipa_a)), len(normalize_ipa(ipa_b)), 1)


def decide_match(
    query: str,
    candidates: Mapping[str, str],
    distance_function: Callable[[str, str], float],
    *,
    temperature: float = 0.5,
    max_normalized_distance: float = 0.8,
    min_margin: float = 0.08,
    min_confidence: float = 0.8,
) -> MatchResult:
    """Score an IPA query and reject weak or ambiguous matches."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    query = normalize_ipa(query)
    raw_scores: list[tuple[str, str, float, float]] = []
    for word, candidate_ipa in candidates.items():
        candidate_ipa = normalize_ipa(candidate_ipa)
        distance = normalized_distance(
            distance_function(query, candidate_ipa), query, candidate_ipa
        )
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("distance_function must return a finite non-negative value")
        raw_scores.append((word, candidate_ipa, distance, -distance / temperature))

    raw_scores.sort(key=lambda score: score[2])
    highest_log_score = max(score[3] for score in raw_scores)
    denominator = sum(math.exp(score[3] - highest_log_score) for score in raw_scores)
    scores = [
        CandidateScore(word, ipa, distance, math.exp(log_score - highest_log_score) / denominator)
        for word, ipa, distance, log_score in raw_scores
    ]

    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    margin = second.distance - best.distance if second else math.inf
    accepted = (
        best.distance <= max_normalized_distance
        and margin >= min_margin
        and best.confidence >= min_confidence
    )
    return MatchResult("likely_match" if accepted else "ambiguous_or_unknown", best, second, margin)


def record_word(seconds: float = 2.0) -> Path:
    """Record mono 16-bit audio to a temporary WAV file."""
    if seconds <= 0:
        raise ValueError("recording duration must be greater than zero")

    import sounddevice as sd

    device_info = sd.query_devices(kind="input")
    sample_rate = int(device_info["default_samplerate"])
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()

    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    path = Path(temporary.name)
    try:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def speech_to_ipa(wav_path: Union[str, Path]) -> str:
    """Transcribe a WAV file using an Allosaurus recognizer."""
    return normalize_ipa(_recognizer().recognize(str(wav_path), lang_id="ipa"))


@lru_cache(maxsize=1)
def _recognizer():
    from allosaurus.app import read_recognizer

    return read_recognizer()


def phonetic_distance(a: str, b: str) -> float:
    return _distance_calculator().weighted_feature_edit_distance(
        normalize_ipa(a), normalize_ipa(b)
    )


@lru_cache(maxsize=1)
def _distance_calculator():
    from panphon.distance import Distance

    return Distance()


def listen_and_match(
    words: Mapping[str, str] = DEFAULT_WORDS,
    *,
    seconds: float = 2.0,
) -> tuple[str, MatchResult]:
    """Record speech, transcribe it, score it, and always clean up the recording."""
    wav_path = record_word(seconds)
    try:
        ipa = speech_to_ipa(wav_path)
        return ipa, decide_match(ipa, words, phonetic_distance)
    finally:
        wav_path.unlink(missing_ok=True)
