"""IPA text normalization shared by recognition and matching."""

import unicodedata

# These are notation aliases, not merely similar sounds. Distinct phonemes such
# as /r/ and /ɾ/ deliberately remain separate.
_IPA_EQUIVALENTS = str.maketrans(
    {
        "g": "ɡ",
        ":": "ː",
        "꞉": "ː",
        "'": "ʼ",
        "‘": "ʼ",
        "’": "ʼ",
        "ʹ": "ʼ",
        "′": "ʼ",
        "͜": "͡",
    }
)


def normalize_ipa(ipa: str) -> str:
    """Remove separators and canonicalize equivalent IPA notation."""
    compact = "".join(ipa.split()).translate(_IPA_EQUIVALENTS)
    return unicodedata.normalize("NFC", compact)
