import unittest

from phonomatch import DEFAULT_WORDS, phones_for_words
from phonomatch.infrastructure.phonetics import (
    phone_sequences_for_words,
    phonetic_distance,
    phonetic_maximum_distance,
)


class PhoneticsTests(unittest.TestCase):
    def test_phone_inventory_is_derived_from_all_words(self) -> None:
        phones = phones_for_words(DEFAULT_WORDS)
        self.assertIn("ɡ", phones)
        self.assertIn("aː", phones)
        self.assertIn("ʃ", phones)
        self.assertNotIn("ˈ", phones)

    def test_panphon_distance_is_bounded_by_maximum(self) -> None:
        distance = phonetic_distance("tova", "naku")
        maximum = phonetic_maximum_distance("tova", "naku")
        self.assertGreaterEqual(distance, 0)
        self.assertLessEqual(distance, maximum)

    def test_phone_sequences_are_derived_for_each_vocabulary_word(self) -> None:
        sequences = phone_sequences_for_words({"grun": "ɡrun", "shaki": "ʃaki"})
        self.assertEqual(sequences["grun"], ("ɡ", "r", "u", "n"))
        self.assertEqual(sequences["shaki"], ("ʃ", "a", "k", "i"))

    def test_phone_sequences_reject_unrecognized_pronunciations(self) -> None:
        with self.assertRaisesRegex(ValueError, "no recognized IPA phones"):
            phone_sequences_for_words({"invalid": "   "})
