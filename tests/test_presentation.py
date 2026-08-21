import unittest
from dataclasses import replace

from sound_analyzer import (
    MatchSettings,
    PhraseAnalysisResult,
    SoundAnalyzer,
    WordAnalysisResult,
)
from sound_analyzer.interfaces.console import render_phrase_result, render_result


class PresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MatchSettings()
        self.result = SoundAnalyzer(
            {"grun": "ɡrun", "naku": "naku"}, self.settings
        ).match_ipa("gɾun")

    def test_report_contains_candidates_and_checks(self) -> None:
        report = render_result(self.result, self.settings)
        self.assertIn("✓ LIKELY MATCH", report)
        self.assertIn("Candidates", report)
        self.assertIn("Relative separation", report)
        self.assertIn("PASS", report)

    def test_color_output_uses_ansi_sequences(self) -> None:
        report = render_result(self.result, self.settings, color=True)
        self.assertIn("\033[", report)

    def test_report_displays_recognition_duration(self) -> None:
        result = replace(self.result, recognition_seconds=1.234)
        report = render_result(result, self.settings)
        self.assertIn("Time     1.23 s", report)

    def test_phrase_report_contains_word_results_and_timestamps(self) -> None:
        phrase = PhraseAnalysisResult(
            words=(WordAnalysisResult("gɾun", self.result.match, 0.12, 0.74),),
            sequence_confidence=0.92,
            sequence_accepted=True,
            alternative_words=("naku",),
        )

        report = render_phrase_result(phrase, self.settings)

        self.assertIn("PHRASE ACCEPTED", report)
        self.assertIn("Sequence confidence  92.0%", report)
        self.assertIn("0.12–0.74s", report)
        self.assertIn("Runner-up  naku", report)
