"""Backward-compatible entry point for running the project from the repository root."""

from phonomatch.interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
