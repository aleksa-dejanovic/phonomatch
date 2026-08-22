import unittest

from phonomatch import DEFAULT_WORDS, phones_for_words
from phonomatch.infrastructure.phonetics import (
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
