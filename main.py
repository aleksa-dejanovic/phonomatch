"""Backward-compatible entry point for running the project from the repository root."""

from sound_analyzer import main


if __name__ == "__main__":
    raise SystemExit(main())
