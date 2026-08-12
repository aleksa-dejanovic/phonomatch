"""Microphone capture and temporary WAV lifecycle management."""

from __future__ import annotations

import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exceptions import AudioRecordingError


@contextmanager
def recorded_audio(seconds: float = 2.0) -> Iterator[Path]:
    """Record mono 16-bit audio and remove the temporary WAV on exit."""
    if seconds <= 0:
        raise ValueError("recording duration must be greater than zero")

    path: Path | None = None
    try:
        path = _record_to_wav(seconds)
        yield path
    except (AudioRecordingError, ValueError):
        raise
    except Exception as exc:
        raise AudioRecordingError(f"could not record audio: {exc}") from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _record_to_wav(seconds: float) -> Path:
    import sounddevice as sd

    try:
        device_info: Any = sd.query_devices(kind="input")
        sample_rate = int(device_info["default_samplerate"])
        audio: Any = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as exc:
        raise AudioRecordingError(f"microphone capture failed: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
