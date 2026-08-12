import unittest

from sound_analyzer import SoundAnalyzer


class AnalyzerTests(unittest.TestCase):
    def test_empty_vocabulary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            SoundAnalyzer({})

    def test_match_ipa_returns_analysis_result(self) -> None:
        result = SoundAnalyzer({"grun": "ɡrun", "naku": "naku"}).match_ipa("gɾun")
        self.assertEqual(result.recognized_ipa, "gɾun")
        self.assertTrue(result.match.accepted)
        self.assertEqual(result.match.best_candidate.word, "grun")
