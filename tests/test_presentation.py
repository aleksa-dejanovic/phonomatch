import unittest
from dataclasses import replace

from sound_analyzer import MatchSettings, SoundAnalyzer
from sound_analyzer.presentation import render_result


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
