"""Allosaurus integration with vocabulary-specific phone masking."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from .exceptions import RecognitionError, UnsupportedPhoneError
from .ipa import normalize_ipa


def speech_to_ipa(
    wav_path: Union[str, Path],
    allowed_phones: Optional[Iterable[str]] = None,
) -> str:
    """Transcribe a WAV file, optionally restricting the decoder inventory."""
    path = Path(wav_path)
    if not path.is_file():
        raise RecognitionError(f"audio file does not exist: {path}")
    if path.suffix.lower() != ".wav":
        raise RecognitionError("Allosaurus only supports WAV input")

    recognizer = _recognizer()
    try:
        if allowed_phones is None:
            transcription = recognizer.recognize(str(path), lang_id="ipa")
        else:
            transcription = _recognize_with_inventory(recognizer, path, allowed_phones)
    except (RecognitionError, UnsupportedPhoneError):
        raise
    except Exception as exc:
        raise RecognitionError(f"speech recognition failed: {exc}") from exc
    return normalize_ipa(str(transcription))


def _recognize_with_inventory(
    recognizer: Any,
    wav_path: Path,
    allowed_phones: Iterable[str],
) -> str:
    phones = tuple(dict.fromkeys(normalize_ipa(phone) for phone in allowed_phones))
    if not phones:
        raise ValueError("allowed_phones must contain at least one phone")

    model_inventory = recognizer.lm.inventory.unit
    unsupported = [phone for phone in phones if phone not in model_inventory]
    if unsupported:
        raise UnsupportedPhoneError(
            "phones unsupported by the Allosaurus model: " + ", ".join(unsupported)
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".phones", encoding="utf-8", delete=False
    ) as inventory_file:
        inventory_path = Path(inventory_file.name)
        inventory_file.write("\n".join(phones) + "\n")
    try:
        return str(recognizer.recognize(str(wav_path), lang_id=str(inventory_path)))
    finally:
        inventory_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _recognizer() -> Any:
    from allosaurus.app import read_recognizer

    return read_recognizer()
