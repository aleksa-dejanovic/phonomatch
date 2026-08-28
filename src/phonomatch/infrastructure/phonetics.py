"""PanPhon-backed distance and inventory operations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
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


@dataclass(frozen=True)
class Vocabulary:
    """A validated vocabulary and its PanPhon-derived phone representations."""

    words: Mapping[str, str]
    phone_sequences: Mapping[str, tuple[str, ...]]
    phones: tuple[str, ...]

    @classmethod
    def from_words(cls, words: Mapping[Any, Any]) -> Vocabulary:
        """Validate a vocabulary and derive its normalized phone sequences."""
        if not words:
            raise ValueError("at least one vocabulary word is required")
        feature_table = _distance_calculator().fm
        validated_words: dict[str, str] = {}
        sequences: dict[str, tuple[str, ...]] = {}
        for word, ipa in words.items():
            if not isinstance(word, str) or not word.strip():
                raise ValueError("vocabulary words must be non-empty strings")
            if not isinstance(ipa, str):
                raise ValueError(
                    f"vocabulary pronunciation for {word!r} must be a string"
                )
            normalized_ipa = normalize_ipa(ipa)
            phones = tuple(feature_table.ipa_segs(normalized_ipa))
            if not phones:
                raise ValueError(
                    "vocabulary pronunciation for "
                    f"{word!r} contains no recognized IPA phones"
                )
            validated_words[word] = ipa
            sequences[word] = phones
        return cls(
            words=MappingProxyType(validated_words),
            phone_sequences=MappingProxyType(sequences),
            phones=tuple(
                sorted({phone for sequence in sequences.values() for phone in sequence})
            ),
        )


@lru_cache(maxsize=1)
def _distance_calculator() -> Any:
    # PanPhon 0.22 emits this warning while importing its generated segment
    # definitions. Keep it local so unrelated deprecation warnings are visible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence .*",
            category=DeprecationWarning,
        )
        from panphon.distance import Distance

    return Distance()
