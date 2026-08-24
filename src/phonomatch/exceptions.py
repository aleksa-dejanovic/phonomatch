"""Exceptions raised by phonomatch integrations."""


class PhonoMatchError(Exception):
    """Base class for expected phonomatch failures."""


class OptionalDependencyError(PhonoMatchError):
    """Raised when an operation needs an optional installation extra."""

    def __init__(self, extra: str) -> None:
        super().__init__(
            f"this feature requires the {extra!r} extra; install it with "
            f"python -m pip install 'phonomatch[{extra}]'"
        )


class AudioRecordingError(PhonoMatchError):
    """Raised when microphone capture fails."""


class RecognitionError(PhonoMatchError):
    """Raised when Wav2Vec2Phoneme cannot transcribe a recording."""


class UnsupportedPhoneError(RecognitionError):
    """Raised when a vocabulary phone is unavailable in Wav2Vec2Phoneme."""
