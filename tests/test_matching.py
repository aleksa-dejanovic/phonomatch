import math
import unittest

from phonematch import MatchSettings, decide_match, distance_ratio


def simple_distance(a: str, b: str) -> float:
    substitutions = sum(left != right for left, right in zip(a, b, strict=False))
    return float(substitutions + abs(len(a) - len(b)))


class MatchingTests(unittest.TestCase):
    def test_distance_ratio_uses_maximum_cost(self) -> None:
        self.assertEqual(distance_ratio(0, 0), 0)
        self.assertEqual(distance_ratio(5, 100), 0.05)

    def test_clear_match_is_accepted(self) -> None:
        result = decide_match("tova", {"tova": "tova", "naku": "naku"}, simple_distance)
        self.assertTrue(result.accepted)
        self.assertEqual(result.best_candidate.word, "tova")
        self.assertGreater(result.relative_margin, 0)

    def test_equivalent_g_symbols_match(self) -> None:
        result = decide_match("grun", {"grun": "ɡrun", "naku": "naku"}, simple_distance)
        self.assertTrue(result.accepted)
        self.assertEqual(result.best_candidate.distance_ratio, 0)

    def test_ambiguous_match_is_rejected(self) -> None:
        result = decide_match("a", {"first": "a", "second": "a"}, simple_distance)
        self.assertFalse(result.accepted)
        self.assertEqual(result.relative_margin, 0)

    def test_relative_margin_accounts_for_candidate_similarity(self) -> None:
        result = decide_match(
            "aaaa", {"best": "aaaa", "similar": "aaab"}, simple_distance
        )
        self.assertEqual(result.relative_margin, 1)

    def test_identical_candidate_pronunciations_are_ambiguous(self) -> None:
        result = decide_match(
            "aaaa", {"first": "aaaa", "second": "aaaa"}, simple_distance
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.relative_margin, 0)

    def test_empty_candidates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            decide_match("a", {}, simple_distance)

    def test_invalid_distance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            decide_match("a", {"a": "a"}, lambda _a, _b: math.nan)

    def test_invalid_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            MatchSettings(max_distance_ratio=1.1)
