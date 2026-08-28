"""Domain defaults and validated matching configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

DEFAULT_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "naku": "naku",
        "selim": "selim",
        "tova": "tova",
        "grun": "ɡrun",
        "shaki": "ʃaki",
        "flabkiver": "flæbˈkɪvər",
        "dracarys": "draˈkarys",
        "lykiri": "lyˈkiri",
        "dohaeras": "dohaeˈra:s",
        "umbas": "ˈʊmbæs",
        "rybas": "ˈriːbəs",
        "mazis": "mˈæ.ziz",
        "naejot": "ˈnaeɟot",
        "soves": "ˈsuːvɛs",
        "vezos": "ˈˈveːzos",
        "kepus": "ˈkɛpus",
        "sovetes": "soˈvetes",
        "drakaryssy": "drakaˈryssy",
    }
)


@dataclass(frozen=True)
class MatchSettings:
    """Thresholds used to accept or reject the best phonetic candidate."""

    temperature: float = 0.035
    max_distance_ratio: float = 0.1
    min_relative_margin: float = 0.20
    min_confidence: float = 0.80

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a finite value greater than zero")
        for name in (
            "max_distance_ratio",
            "min_relative_margin",
            "min_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite value between zero and one")


DEFAULT_SETTINGS = MatchSettings()
