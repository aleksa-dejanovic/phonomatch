"""Pure functions for ranking and accepting phonetic candidates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from .config import DEFAULT_SETTINGS, MatchSettings
from .ipa import normalize_ipa
from .models import CandidateScore, Decision, MatchResult

DistanceFunction = Callable[[str, str], float]


def distance_ratio(cost: float, maximum_cost: float) -> float:
    """Express an edit cost as a fraction of its theoretical maximum."""
    if cost < 0 or not math.isfinite(cost):
        raise ValueError("distance function must return a finite non-negative value")
    if maximum_cost < 0 or not math.isfinite(maximum_cost):
        raise ValueError(
            "maximum distance function must return a finite non-negative value"
        )
    if maximum_cost == 0:
        return 0.0 if cost == 0 else math.inf
    if cost > maximum_cost:
        raise ValueError("distance cost cannot exceed its stated maximum")
    return cost / maximum_cost


def unit_edit_maximum(ipa_a: str, ipa_b: str) -> float:
    """Return the delete-all/insert-all bound for unit-cost edit distance."""
    return float(len(ipa_a) + len(ipa_b))


def decide_match(
    query: str,
    candidates: Mapping[str, str],
    distance_function: DistanceFunction,
    *,
    maximum_distance_function: DistanceFunction = unit_edit_maximum,
    settings: MatchSettings = DEFAULT_SETTINGS,
) -> MatchResult:
    """Rank IPA candidates and reject distant or ambiguous results."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    normalized_query = normalize_ipa(query)
    raw_scores: list[tuple[str, str, float, float, float]] = []
    for word, ipa in candidates.items():
        normalized_ipa = normalize_ipa(ipa)
        raw_distance = distance_function(normalized_query, normalized_ipa)
        ratio = distance_ratio(
            raw_distance,
            maximum_distance_function(normalized_query, normalized_ipa),
        )
        raw_scores.append(
            (
                word,
                normalized_ipa,
                raw_distance,
                ratio,
                -ratio / settings.temperature,
            )
        )

    # Ratios make candidate ranking comparable across different word lengths.
    raw_scores.sort(key=lambda score: score[3])
    highest_log_score = max(score[4] for score in raw_scores)
    denominator = sum(math.exp(score[4] - highest_log_score) for score in raw_scores)
    scores = [
        CandidateScore(
            word=word,
            ipa=ipa,
            raw_distance=raw_distance,
            distance_ratio=ratio,
            confidence=math.exp(log_score - highest_log_score) / denominator,
        )
        for word, ipa, raw_distance, ratio, log_score in raw_scores
    ]

    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    relative_margin = _relative_margin(best, second, distance_function)
    accepted = (
        best.distance_ratio <= settings.max_distance_ratio
        and relative_margin >= settings.min_relative_margin
        and best.confidence >= settings.min_confidence
    )
    decision: Decision = "likely_match" if accepted else "ambiguous_or_unknown"
    return MatchResult(decision, best, second, relative_margin)


def _relative_margin(
    best: CandidateScore,
    second: CandidateScore | None,
    distance_function: DistanceFunction,
) -> float:
    if second is None:
        return math.inf

    candidate_distance = distance_function(best.ipa, second.ipa)
    if not math.isfinite(candidate_distance) or candidate_distance < 0:
        raise ValueError("distance function must return a finite non-negative value")
    if candidate_distance == 0:
        # Identical pronunciations cannot be distinguished from audio alone.
        return 0.0

    margin = (second.raw_distance - best.raw_distance) / candidate_distance
    return max(-1.0, min(1.0, margin))
