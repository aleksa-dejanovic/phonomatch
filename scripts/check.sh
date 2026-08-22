#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "$temporary_directory"' EXIT

usage() {
    echo "Usage: $0"
    echo
    echo "Run formatting, lint, type, test, lockfile, and package-build checks."
}

if (( $# > 0 )); then
    usage >&2
    exit 2
fi

cd "$repository_root"

run_check() {
    local label="$1"
    shift
    printf '\n==> %s\n' "$label"
    "$@"
}

run_check "Checking patch whitespace" git diff --check
run_check "Checking the lockfile" uv lock --check
run_check "Checking formatting" uv run ruff format --check .
run_check "Running lint" uv run ruff check .
run_check "Running strict type checks" uv run --group dev python -m mypy
run_check \
    "Running unit tests with coverage" \
    uv run --group dev coverage run -m unittest discover -s tests -v
run_check "Reporting coverage" uv run --group dev coverage report
run_check "Generating HTML coverage report" uv run --group dev coverage html
printf '\nHTML coverage report: %s\n' "$repository_root/htmlcov/index.html"
run_check \
    "Building source and wheel distributions" \
    uv build --out-dir "$temporary_directory/dist"

printf '\nAll checks passed.\n'
