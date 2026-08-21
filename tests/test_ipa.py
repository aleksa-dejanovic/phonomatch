import unittest

from phonematch import normalize_ipa


class IpaTests(unittest.TestCase):
    def test_removes_all_whitespace(self) -> None:
        self.assertEqual(normalize_ipa("  ʃ a\n k i "), "ʃaki")

    def test_converts_notation_aliases(self) -> None:
        self.assertEqual(normalize_ipa("gɾun"), "ɡɾun")
        self.assertEqual(normalize_ipa("a: t’s"), "aːtʼs")
        self.assertEqual(normalize_ipa("t͜ʃ"), "t͡ʃ")
