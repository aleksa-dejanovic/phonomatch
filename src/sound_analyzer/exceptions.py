"""Exceptions raised by sound-analyzer integrations."""


class SoundAnalyzerError(Exception):
    """Base class for expected sound-analyzer failures."""


class AudioRecordingError(SoundAnalyzerError):
    """Raised when microphone capture fails."""


class RecognitionError(SoundAnalyzerError):
    """Raised when Wav2Vec2Phoneme cannot transcribe a recording."""


class UnsupportedPhoneError(RecognitionError):
    """Raised when a vocabulary phone is unavailable in Wav2Vec2Phoneme."""
