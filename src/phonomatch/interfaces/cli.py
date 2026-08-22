"""Command-line interface for phonomatch."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import Optional

from ..application.analyzer import PhonoMatch
from ..domain.config import MatchSettings
from ..exceptions import PhonoMatchError
from .console import Palette, color_enabled, render_phrase_result, render_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phonomatch",
        description="Match a spoken word using its IPA transcription.",
    )
    parser.add_argument(
        "--phrase",
        action="store_true",
        help="recognize a sequence of vocabulary words",
    )
    parser.add_argument(
        "--beam-size",
        type=_positive_int,
        default=64,
        help="phrase decoder beam size (default: 64)",
    )
    parser.add_argument(
        "--max-words",
        type=_positive_int,
        default=12,
        help="maximum words in phrase mode (default: 12)",
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
        analyzer = PhonoMatch(settings=settings)
        print(palette.cyan("◌ Loading speech model..."), flush=True)
        analyzer.load_model()
        print(palette.green("● Model ready"), flush=True)
        prompt = (
            "● Listening — say a phrase..."
            if args.phrase
            else "● Listening — say a word..."
        )
        print(palette.cyan(prompt), flush=True)
        if args.phrase:
            phrase_result = analyzer.listen_for_phrase(
                seconds=args.seconds,
                beam_size=args.beam_size,
                max_words=args.max_words,
                on_recording_complete=recording_complete,
            )
            print()
            print(render_phrase_result(phrase_result, settings, color=use_color))
            return 0 if phrase_result.accepted else 2
        else:
            word_result = analyzer.listen(
                seconds=args.seconds,
                on_recording_complete=recording_complete,
            )
            print()
            print(render_result(word_result, settings, color=use_color))
            return 0 if word_result.match.accepted else 2
    except (PhonoMatchError, ValueError) as exc:
        error_palette = Palette(color_enabled(args.color, sys.stderr))
        print(error_palette.red(f"Error: {exc}"), file=sys.stderr)
        return 1


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
