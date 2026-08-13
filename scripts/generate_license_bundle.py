"""Generate release notices from the installed locked runtime environment."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import deque
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT = "sound-analyzer"
LICENSE_NAMES = ("license", "licence", "copying", "notice", "copyright")
STRONG_COPYLEFT = re.compile(r"(?<!L)GPL(?:-|\s|$)|AGPL", re.IGNORECASE)


def runtime_distributions() -> list[Distribution]:
    """Resolve the installed runtime dependency closure from package metadata."""
    queue = deque([PROJECT])
    seen: set[str] = set()
    result: list[Distribution] = []

    while queue:
        name = canonicalize_name(queue.popleft())
        if name in seen:
            continue
        seen.add(name)
        try:
            package = distribution(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"runtime dependency is not installed: {name}") from exc
        if name != PROJECT:
            result.append(package)

        for requirement_text in package.requires or ():
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            queue.append(requirement.name)

    return sorted(result, key=lambda item: canonicalize_name(item.metadata["Name"]))


def declared_license(package: Distribution) -> str:
    """Return concise license metadata without embedding full license bodies."""
    value = package.metadata.get("License-Expression") or package.metadata.get(
        "License"
    )
    if value and len(value) <= 160 and "\n" not in value:
        return value
    classifiers = package.metadata.get_all("Classifier") or ()
    licenses = [
        item.rsplit("::", 1)[-1].strip() for item in classifiers if "License ::" in item
    ]
    return " OR ".join(licenses) if licenses else "See bundled license files"


def license_files(package: Distribution) -> list[Path]:
    result: list[Path] = []
    for entry in package.files or ():
        basename = Path(str(entry)).name.lower()
        if any(basename.startswith(prefix) for prefix in LICENSE_NAMES):
            source = Path(package.locate_file(entry))
            if source.is_file():
                result.append(source)
    return sorted(set(result))


def generate(output: Path) -> None:
    packages = runtime_distributions()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rows: list[tuple[str, str, str, str]] = []
    prohibited: list[str] = []
    missing: list[str] = []
    apache_source: Path | None = None

    for package in packages:
        name = package.metadata["Name"]
        license_name = declared_license(package)
        sources = license_files(package)
        if canonicalize_name(name) == "transformers" and sources:
            apache_source = sources[0]
        if STRONG_COPYLEFT.search(license_name):
            prohibited.append(f"{name} {package.version}: {license_name}")
        if not sources and canonicalize_name(name) not in {"tokenizers", "unicodecsv"}:
            missing.append(f"{name} {package.version}")

        package_dir = output / f"{canonicalize_name(name)}-{package.version}"
        package_dir.mkdir()
        copied: list[str] = []
        for index, source in enumerate(sources, start=1):
            filename = (
                source.name if source.name not in copied else f"{index}-{source.name}"
            )
            shutil.copyfile(source, package_dir / filename)
            copied.append(filename)
        rows.append((name, package.version, license_name, ", ".join(copied)))

    if apache_source is None:
        raise RuntimeError("Transformers Apache license could not be located")

    # tokenizers 0.22.2 omits its license file from the wheel. Its tagged source
    # is Apache-2.0, so ship the identical Apache text already provided by
    # Transformers and call out the source in the report.
    tokenizers_row = next(
        row for row in rows if canonicalize_name(row[0]) == "tokenizers"
    )
    tokenizers_dir = output / f"tokenizers-{tokenizers_row[1]}"
    shutil.copyfile(apache_source, tokenizers_dir / "LICENSE-APACHE-2.0")
    rows = [
        (name, version, "Apache-2.0", "LICENSE-APACHE-2.0")
        if canonicalize_name(name) == "tokenizers"
        else (name, version, license_name, files)
        for name, version, license_name, files in rows
    ]

    # unicodecsv's upstream sdist contains no license file. Preserve the
    # publisher's license metadata and origin instead of inventing terms.
    unicodecsv_row = next(
        row for row in rows if canonicalize_name(row[0]) == "unicodecsv"
    )
    unicodecsv_dir = output / f"unicodecsv-{unicodecsv_row[1]}"
    (unicodecsv_dir / "LICENSE-METADATA.txt").write_text(
        "unicodecsv 0.14.1\n"
        "Author: Jeremy Dunck\n"
        "Source: https://pypi.org/project/unicodecsv/0.14.1/\n"
        "Upstream metadata: BSD License; OSI Approved :: BSD License\n"
        "The upstream source distribution does not include a license text.\n",
        encoding="utf-8",
    )
    rows = [
        (name, version, license_name, "LICENSE-METADATA.txt")
        if canonicalize_name(name) == "unicodecsv"
        else (name, version, license_name, files)
        for name, version, license_name, files in rows
    ]

    model_dir = output / "model-facebook-wav2vec2-lv-60-espeak-cv-ft"
    model_dir.mkdir()
    shutil.copyfile(apache_source, model_dir / "LICENSE-APACHE-2.0")

    if missing:
        raise RuntimeError("missing license files for: " + ", ".join(missing))
    if prohibited:
        raise RuntimeError(
            "strong-copyleft runtime package found: " + ", ".join(prohibited)
        )

    report = [
        "# Runtime dependency license report",
        "",
        "Generated from the installed `uv.lock` environment. Re-run this script",
        "for every target-platform release artifact; dependency sets are "
        "platform-specific.",
        "",
        "| Component | Version | Declared license | Bundled files |",
        "| --- | --- | --- | --- |",
    ]
    report.extend(
        f"| {name} | {version} | {license_name} | {files} |"
        for name, version, license_name, files in rows
    )
    report.extend(
        (
            "| facebook/wav2vec2-lv-60-espeak-cv-ft | "
            "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4 | Apache-2.0 | "
            "LICENSE-APACHE-2.0 |",
            "",
        )
    )
    Path("DEPENDENCY_LICENSE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("THIRD_PARTY_LICENSES"))
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
