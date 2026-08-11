# pip install allosaurus panphon sounddevice scipy

from allosaurus.app import read_recognizer
from panphon.distance import Distance
from scipy.io.wavfile import write
import sounddevice as sd
import tempfile
import unicodedata

device_info = sd.query_devices(kind="input")

SAMPLE_RATE = int(device_info["default_samplerate"])
RECORD_SECONDS = 2.0

# Predefined made-up words and their intended IPA pronunciations.
WORDS = {
    "naku": "naku",
    "selim": "selim",
    "tova": "tova",
    "grun": "ɡrun",
    "shaki": "ʃaki",
    "flabkiver": "flæbˈkɪvər"
}

# Tune these experimentally.
MAX_DISTANCE = 5
MIN_MARGIN = 0.4


recognizer = read_recognizer()
distance = Distance()


def record_word(seconds=RECORD_SECONDS):
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ).name

    write(path, SAMPLE_RATE, audio)

    return path


def normalize_ipa(ipa):
    # Allosaurus returns phones separated by spaces.
    ipa = ipa.replace(" ", "")

    return unicodedata.normalize("NFC", ipa)


def speech_to_ipa(wav_path):
    phones = recognizer.recognize(
        wav_path,
        lang_id="ipa",
    )

    return normalize_ipa(phones)


def phonetic_distance(a, b):
    return distance.weighted_feature_edit_distance(
        normalize_ipa(a),
        normalize_ipa(b),
    )


def find_word(recognized_ipa):
    candidates = []

    for word, expected_ipa in WORDS.items():
        score = phonetic_distance(
            recognized_ipa,
            expected_ipa,
        )

        candidates.append(
            (score, word, expected_ipa)
        )

    candidates.sort()

    best_score, best_word, best_ipa = candidates[0]

    second_score = (
        candidates[1][0]
        if len(candidates) > 1
        else float("inf")
    )

    margin = second_score - best_score

    accepted = (
        best_score <= MAX_DISTANCE
        and margin >= MIN_MARGIN
    )

    return {
        "accepted": accepted,
        "word": best_word if accepted else None,
        "recognized_ipa": recognized_ipa,
        "expected_ipa": best_ipa,
        "distance": best_score,
        "margin": margin,
        "candidates": candidates,
    }


def listen_and_match():
    wav_path = record_word()

    ipa = speech_to_ipa(wav_path)

    return find_word(ipa)


if __name__ == "__main__":
    print("Say a word...")

    result = listen_and_match()

    print("Recognized IPA:", result["recognized_ipa"])

    print("\nCandidates:")

    for score, word, ipa in result["candidates"]:
        print(
            f"{word:10} "
            f"/{ipa}/ "
            f"distance={score:.3f}"
        )

    if result["accepted"]:
        print(
            "\nMATCH:",
            result["word"],
            f"(distance={result['distance']:.3f})",
        )
    else:
        print("\nUNKNOWN")
