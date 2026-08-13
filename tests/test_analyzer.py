import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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

    def test_listen_reports_when_recording_is_complete(self) -> None:
        analyzer = SoundAnalyzer({"grun": "ɡrun", "naku": "naku"})
        events: list[str] = []

        @contextmanager
        def fake_recording(_seconds: float) -> Iterator[Path]:
            yield Path("recording.wav")

        expected = analyzer.match_ipa("gɾun")
        with (
            patch("sound_analyzer.analyzer.recorded_audio", fake_recording),
            patch.object(analyzer, "analyze_file", return_value=expected),
        ):
            result = analyzer.listen(
                on_recording_complete=lambda: events.append("recording_complete")
            )

        self.assertIs(result, expected)
        self.assertEqual(events, ["recording_complete"])
