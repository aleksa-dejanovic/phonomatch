"""Command-line interface for phonomatch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from ..application.analyzer import PhonoMatch
from ..domain.config import MatchSettings
from ..exceptions import PhonoMatchError
from .console import Palette, color_enabled, render_phrase_result, render_result
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phonomatch",
        description="Match a spoken word using its IPA transcription.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="run a local HTTP recognition service",
    )
    vocabulary = parser.add_mutually_exclusive_group()
    vocabulary.add_argument(
        "--words",
        type=Path,
        metavar="PATH",
        help="JSON file mapping each word to its IPA pronunciation",
    )
    vocabulary.add_argument(
        "--default-words",
        action="store_true",
        help="use PhonoMatch's built-in demonstration vocabulary",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="server bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8765,
        help="server port (default: 8765)",
    )
    parser.add_argument(
        "--enable-model-lifecycle",
        action="store_true",
        help="enable server endpoints that load or unload the speech model",
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
        if args.words is None and not args.default_words:
            raise ValueError(
                "a word list is required; use --words PATH or --default-words"
            )
        words = _read_words(args.words) if args.words is not None else None
        analyzer = PhonoMatch(
            words,
            settings=settings,
            default_words=args.default_words,
        )
        print(palette.cyan("◌ Loading speech model..."), flush=True)
        analyzer.load_model()
        print(palette.green("● Model ready"), flush=True)
        if args.server:
            print(
                palette.cyan(
                    f"● Serving recognition at http://{args.host}:{args.port}"
                ),
                flush=True,
            )
            serve(
                args.host,
                args.port,
                analyzer,
                enable_model_lifecycle=args.enable_model_lifecycle,
            )
            return 0
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


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _read_words(path: Path) -> dict[str, str]:
    """Load a vocabulary from a JSON object of word-to-IPA entries."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read vocabulary file {path}: {exc}") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(word, str) and isinstance(ipa, str) for word, ipa in payload.items()
    ):
        raise ValueError(
            "vocabulary JSON must be an object mapping words to IPA strings"
        )
    return payload
