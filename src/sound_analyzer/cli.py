"""Command-line interface for sound-analyzer."""

import argparse
from collections.abc import Sequence
from typing import Optional

from .analyzer import listen_and_match


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Match a spoken word using its IPA transcription.")
    parser.add_argument("--seconds", type=float, default=2.0, help="recording duration (default: 2)")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")

    print("Say a word...")
    try:
        ipa, result = listen_and_match(seconds=args.seconds)
    except Exception as exc:
        parser.exit(1, f"sound-analyzer: {exc}\n")

    print(f"Recognized IPA: /{ipa}/")
    print(f"Decision: {result.decision}")
    print(
        f"Best candidate: {result.best_candidate.word} /{result.best_candidate.ipa}/ "
        f"(distance={result.best_candidate.distance:.1%} of maximum, "
        f"confidence={result.best_candidate.confidence:.1%})"
    )
    return 0 if result.accepted else 2
