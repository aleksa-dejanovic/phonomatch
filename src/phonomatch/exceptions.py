"""Exceptions raised by phonomatch integrations."""


class PhonoMatchError(Exception):
    """Base class for expected phonomatch failures."""


class AudioRecordingError(PhonoMatchError):
    """Raised when microphone capture fails."""


class RecognitionError(PhonoMatchError):
    """Raised when Wav2Vec2Phoneme cannot transcribe a recording."""


class UnsupportedPhoneError(RecognitionError):
    """Raised when a vocabulary phone is unavailable in Wav2Vec2Phoneme."""
