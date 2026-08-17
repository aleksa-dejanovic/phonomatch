"""PanPhon-backed distance and inventory operations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ..domain.ipa import normalize_ipa


def phonetic_distance(ipa_a: str, ipa_b: str) -> float:
    """Return PanPhon's weighted feature edit distance."""
    value = _distance_calculator().weighted_feature_edit_distance(
        normalize_ipa(ipa_a), normalize_ipa(ipa_b)
    )
    return float(value)


def phonetic_maximum_distance(ipa_a: str, ipa_b: str) -> float:
    """Return PanPhon's delete-all/insert-all bound for two phone strings."""
    calculator = _distance_calculator()
    source = calculator.fm.word_to_vector_list(normalize_ipa(ipa_a), numeric=True)
    target = calculator.fm.word_to_vector_list(normalize_ipa(ipa_b), numeric=True)
    return float(sum(calculator.fm.weights) * (len(source) + len(target)))


def phones_for_words(words: Mapping[str, str]) -> tuple[str, ...]:
    """Return the unique PanPhon segments used by a vocabulary."""
    feature_table = _distance_calculator().fm
    phones = {
        phone
        for ipa in words.values()
        for phone in feature_table.ipa_segs(normalize_ipa(ipa))
    }
    if not phones:
        raise ValueError("the vocabulary does not contain any recognized IPA phones")
    return tuple(sorted(phones))


@lru_cache(maxsize=1)
def _distance_calculator() -> Any:
    # PanPhon 0.21 imports pkg_resources, which emits a warning even though the
    # compatible Setuptools version is pinned. Keep this suppression local so
    # unrelated dependency and application warnings remain visible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence .*",
            category=DeprecationWarning,
        )
        from panphon.distance import Distance

    return Distance()
