import math
import unittest

from sound_analyzer.analyzer import decide_match, normalize_ipa, normalized_distance


def simple_distance(a: str, b: str) -> float:
    return sum(left != right for left, right in zip(a, b)) + abs(len(a) - len(b))


class AnalyzerTests(unittest.TestCase):
    def test_normalize_ipa_removes_all_whitespace(self):
        self.assertEqual(normalize_ipa("  ʃ a\n k i "), "ʃaki")

    def test_normalize_ipa_converts_latin_g_to_ipa_g(self):
        self.assertEqual(normalize_ipa("gɾun"), "ɡɾun")

    def test_normalize_ipa_converts_notation_aliases(self):
        self.assertEqual(normalize_ipa("a: t’s"), "aːtʼs")
        self.assertEqual(normalize_ipa("t͜ʃ"), "t͡ʃ")

    def test_normalized_distance_handles_empty_strings(self):
        self.assertEqual(normalized_distance(0, "", ""), 0)

    def test_clear_match_is_accepted(self):
        result = decide_match("tova", {"tova": "tova", "naku": "naku"}, simple_distance)
        self.assertTrue(result.accepted)
        self.assertEqual(result.best_candidate.word, "tova")
        self.assertGreater(result.margin, 0)

    def test_equivalent_g_symbols_match(self):
        result = decide_match("grun", {"grun": "ɡrun", "naku": "naku"}, simple_distance)
        self.assertTrue(result.accepted)
        self.assertEqual(result.best_candidate.distance, 0)

    def test_ambiguous_match_is_rejected(self):
        result = decide_match("a", {"first": "a", "second": "a"}, simple_distance)
        self.assertFalse(result.accepted)
        self.assertEqual(result.margin, 0)

    def test_empty_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            decide_match("a", {}, simple_distance)

    def test_invalid_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            decide_match("a", {"a": "a"}, lambda _a, _b: math.nan)


if __name__ == "__main__":
    unittest.main()
