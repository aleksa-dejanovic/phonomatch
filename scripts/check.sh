#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "$temporary_directory"' EXIT

usage() {
    echo "Usage: $0 [--release]"
    echo
    echo "Run formatting, lint, type, test, lockfile, and package-build checks."
    echo "Use --release to also verify generated dependency license artifacts."
}

release_checks=false
case "${1:-}" in
    "") ;;
    --release) release_checks=true ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if (( $# > 1 )); then
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
run_check "Running strict type checks" uv run mypy
run_check \
    "Running unit tests" \
    uv run python -m unittest discover -s tests -v
run_check \
    "Building source and wheel distributions" \
    uv build --out-dir "$temporary_directory/dist"

if "$release_checks"; then
    audit_directory="$temporary_directory/license-audit"
    mkdir -p "$audit_directory"

    printf '\n==> Regenerating dependency license artifacts\n'
    (
        cd "$audit_directory"
        uv run \
            --project "$repository_root" \
            python "$repository_root/scripts/generate_license_bundle.py" \
            --output "$audit_directory/THIRD_PARTY_LICENSES"
    )
    run_check \
        "Checking the dependency license report" \
        diff -u \
        "$repository_root/DEPENDENCY_LICENSE_REPORT.md" \
        "$audit_directory/DEPENDENCY_LICENSE_REPORT.md"
    run_check \
        "Checking bundled dependency licenses" \
        diff -ru \
        "$repository_root/THIRD_PARTY_LICENSES" \
        "$audit_directory/THIRD_PARTY_LICENSES"
fi

printf '\nAll checks passed.\n'
