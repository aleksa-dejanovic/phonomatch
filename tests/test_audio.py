import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from phonomatch.exceptions import AudioRecordingError
from phonomatch.infrastructure.audio import _record_to_wav, recorded_audio


class AudioTests(unittest.TestCase):
    def test_rejects_non_positive_recording_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"), recorded_audio(0):
            pass

    def test_rejects_non_finite_recording_duration(self) -> None:
        for seconds in (float("nan"), float("inf")):
            with (
                self.assertRaisesRegex(ValueError, "finite value"),
                recorded_audio(seconds),
            ):
                pass

    def test_removes_recording_after_context_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"audio")
            with (
                patch(
                    "phonomatch.infrastructure.audio._record_to_wav", return_value=path
                ),
                recorded_audio() as recorded,
            ):
                self.assertEqual(recorded, path)
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())

    def test_wraps_unexpected_recording_errors(self) -> None:
        with (
            patch(
                "phonomatch.infrastructure.audio._record_to_wav",
                side_effect=RuntimeError("device disconnected"),
            ),
            self.assertRaisesRegex(AudioRecordingError, "device disconnected"),
            recorded_audio(),
        ):
            pass

    def test_writes_mono_pcm_wav_from_microphone_data(self) -> None:
        sounddevice = SimpleNamespace(
            query_devices=lambda kind: {"default_samplerate": 8_000},
            rec=lambda *args, **kwargs: np.array([[1], [-2]], dtype=np.int16),
            wait=lambda: None,
        )
        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            path = _record_to_wav(0.5)
        try:
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 8_000)
                self.assertEqual(
                    wav_file.readframes(2), np.array([1, -2], dtype="<i2").tobytes()
                )
        finally:
            path.unlink(missing_ok=True)

    def test_wraps_microphone_capture_errors(self) -> None:
        sounddevice = SimpleNamespace(
            query_devices=lambda kind: (_ for _ in ()).throw(RuntimeError("no input"))
        )
        with (
            patch.dict(sys.modules, {"sounddevice": sounddevice}),
            self.assertRaisesRegex(AudioRecordingError, "no input"),
        ):
            _record_to_wav(1)
