"""Exceptions raised by phonematch integrations."""


class PhoneMatchError(Exception):
    """Base class for expected phonematch failures."""


class AudioRecordingError(PhoneMatchError):
    """Raised when microphone capture fails."""


class RecognitionError(PhoneMatchError):
    """Raised when Wav2Vec2Phoneme cannot transcribe a recording."""


class UnsupportedPhoneError(RecognitionError):
    """Raised when a vocabulary phone is unavailable in Wav2Vec2Phoneme."""
