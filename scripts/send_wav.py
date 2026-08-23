#!/usr/bin/env python3
"""Send a WAV file to a running local PhonoMatch server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_file", type=Path, help="16-bit PCM WAV file to recognize")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="PhonoMatch server URL (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--phrase",
        action="store_true",
        help="use the phrase-recognition endpoint",
    )
    args = parser.parse_args()

    if not args.wav_file.is_file():
        parser.error(f"file does not exist: {args.wav_file}")
    if args.wav_file.suffix.lower() != ".wav":
        parser.error("wav_file must have a .wav extension")

    endpoint = "/v1/recognize/phrase" if args.phrase else "/v1/recognize"
    request = Request(
        args.url.rstrip("/") + endpoint,
        data=args.wav_file.read_bytes(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            payload = json.load(response)
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        print(f"Server returned HTTP {exc.code}: {message}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Could not reach PhonoMatch server: {exc.reason}", file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
