"""Command-line interface for sound-analyzer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Optional

from ..application.analyzer import SoundAnalyzer
from ..domain.config import MatchSettings
from ..exceptions import SoundAnalyzerError
from .console import Palette, color_enabled, render_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound-analyzer",
        description="Match a spoken word using its IPA transcription.",
    )
    parser.add_argument(
        "--seconds",
        type=_positive_float,
        default=2.0,
        help="recording duration (default: 2)",
    )
    parser.add_argument(
        "--max-distance",
        type=_percentage,
        default=MatchSettings().max_distance_ratio,
        metavar="RATIO",
        help="maximum distance ratio from 0 to 1 (default: 0.10)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colored output: auto, always, or never (default: auto)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = MatchSettings(max_distance_ratio=args.max_distance)
    use_color = color_enabled(args.color, sys.stdout)
    palette = Palette(use_color)

    def recording_complete() -> None:
        print(palette.cyan("■ Recording stopped — recognizing..."), flush=True)

    try:
        analyzer = SoundAnalyzer(settings=settings)
        print(palette.cyan("◌ Loading speech model..."), flush=True)
        analyzer.load_model()
        print(palette.green("● Model ready"), flush=True)
        print(palette.cyan("● Listening — say a word..."), flush=True)
        result = analyzer.listen(
            seconds=args.seconds,
            on_recording_complete=recording_complete,
        )
    except (SoundAnalyzerError, ValueError) as exc:
        error_palette = Palette(color_enabled(args.color, sys.stderr))
        print(error_palette.red(f"Error: {exc}"), file=sys.stderr)
        return 1

    print()
    print(render_result(result, settings, color=use_color))
    return 0 if result.match.accepted else 2


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed
