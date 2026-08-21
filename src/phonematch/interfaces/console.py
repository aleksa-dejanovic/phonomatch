"""Terminal-friendly rendering for analysis results."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TextIO

from ..domain.config import MatchSettings
from ..domain.models import AnalysisResult, PhraseAnalysisResult

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Palette:
    """Apply ANSI styles while keeping rendering usable without color."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def green(self, value: str) -> str:
        return self._style(value, "1;32")

    def red(self, value: str) -> str:
        return self._style(value, "1;31")

    def cyan(self, value: str) -> str:
        return self._style(value, "1;36")

    def bold(self, value: str) -> str:
        return self._style(value, "1")

    def dim(self, value: str) -> str:
        return self._style(value, "2")

    def _style(self, value: str, code: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.enabled else value


def color_enabled(mode: str, stream: TextIO) -> bool:
    """Resolve an auto/always/never color preference for a stream."""
    if mode == "always":
        return True
    if mode == "never" or "NO_COLOR" in os.environ:
        return False
    return stream.isatty() and os.environ.get("TERM") != "dumb"


def render_result(
    result: AnalysisResult,
    settings: MatchSettings,
    *,
    color: bool = False,
) -> str:
    """Render candidates and acceptance checks as a readable console report."""
    palette = Palette(color)
    match = result.match
    accepted = match.accepted
    decision = (
        palette.green("✓ LIKELY MATCH") if accepted else palette.red("✗ NO MATCH")
    )

    lines = [
        palette.cyan(palette.bold("PHONEMATCH")),
        palette.dim("=" * 48),
        f"Heard    /{result.recognized_ipa}/",
        f"Result   {decision}",
    ]
    if result.recognition_seconds is not None:
        lines.append(f"Time     {result.recognition_seconds:.2f} s")
    lines.extend(("", palette.bold("Candidates")))

    candidate_rows = [
        (
            "1",
            match.best_candidate.word,
            f"/{match.best_candidate.ipa}/",
            f"{match.best_candidate.distance_ratio:.1%}",
            f"{match.best_candidate.confidence:.1%}",
        )
    ]
    if match.second_candidate is not None:
        second = match.second_candidate
        candidate_rows.append(
            (
                "2",
                second.word,
                f"/{second.ipa}/",
                f"{second.distance_ratio:.1%}",
                f"{second.confidence:.1%}",
            )
        )
    lines.extend(_table(("#", "Word", "IPA", "Distance", "Confidence"), candidate_rows))

    checks = (
        (
            "Distance",
            match.best_candidate.distance_ratio,
            settings.max_distance_ratio,
            "≤",
        ),
        (
            "Relative separation",
            match.relative_margin,
            settings.min_relative_margin,
            "≥",
        ),
        ("Confidence", match.best_candidate.confidence, settings.min_confidence, "≥"),
    )
    check_rows: list[tuple[str, ...]] = []
    for label, observed, required, operator in checks:
        passed = observed <= required if operator == "≤" else observed >= required
        status = "PASS" if passed else "FAIL"
        check_rows.append(
            (label, f"{observed:.1%}", f"{operator} {required:.1%}", status)
        )

    lines.extend(("", palette.bold("Acceptance checks")))
    check_table = _table(("Check", "Observed", "Required", "Status"), check_rows)
    for line in check_table:
        line = line.replace("PASS", palette.green("PASS"))
        line = line.replace("FAIL", palette.red("FAIL"))
        lines.append(line)
    return "\n".join(lines)


def render_phrase_result(
    result: PhraseAnalysisResult,
    settings: MatchSettings,
    *,
    color: bool = False,
) -> str:
    """Render phrase-level and per-word recognition results."""
    palette = Palette(color)
    decision = (
        palette.green("✓ PHRASE ACCEPTED")
        if result.accepted
        else palette.red("✗ PHRASE REJECTED")
    )
    lines = [
        palette.cyan(palette.bold("PHONEMATCH")),
        palette.dim("=" * 72),
        f"Heard    /{result.recognized_ipa}/",
        f"Result   {decision}",
        "Sequence confidence  "
        f"{result.sequence_confidence:.1%} "
        f"(required ≥ {settings.min_confidence:.1%})",
    ]
    if result.recognition_seconds is not None:
        lines.append(f"Time     {result.recognition_seconds:.2f} s")
    if result.alternative_words is not None:
        lines.append(f"Runner-up  {' '.join(result.alternative_words)}")

    rows = []
    for index, word in enumerate(result.words, start=1):
        candidate = word.match.best_candidate
        rows.append(
            (
                str(index),
                candidate.word,
                f"/{word.recognized_ipa}/",
                f"{word.start_seconds:.2f}–{word.end_seconds:.2f}s",
                f"{candidate.distance_ratio:.1%}",
                f"{candidate.confidence:.1%}",
                "PASS" if word.match.accepted else "FAIL",
            )
        )
    lines.extend(("", palette.bold("Words")))
    table = _table(
        ("#", "Word", "IPA", "Time", "Distance", "Confidence", "Status"), rows
    )
    for line in table:
        line = line.replace("PASS", palette.green("PASS"))
        line = line.replace("FAIL", palette.red("FAIL"))
        lines.append(line)
    return "\n".join(lines)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max(_visible_width(value) for value in (header, *(row[i] for row in rows)))
        for i, header in enumerate(headers)
    ]

    def format_row(row: Sequence[str]) -> str:
        cells = (value.ljust(widths[i]) for i, value in enumerate(row))
        return "  ".join(cells).rstrip()

    return [
        format_row(headers),
        format_row(tuple("-" * width for width in widths)),
        *(format_row(row) for row in rows),
    ]


def _visible_width(value: str) -> int:
    return len(_ANSI_RE.sub("", value))
