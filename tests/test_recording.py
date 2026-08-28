import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from phonomatch import MatchSettings, listen_and_match
from phonomatch.application.phonomatch import PhonoMatch


class RecordingWorkflowTests(unittest.TestCase):
    def test_record_and_analyze_word_reports_when_recording_is_complete(self) -> None:
        analyzer = Mock()
        events: list[str] = []

        @contextmanager
        def fake_recording(_seconds: float) -> Generator[Path, None, None]:
            events.append("recording_started")
            yield Path("recording.wav")

        expected = object()
        with (
            patch.object(
                analyzer,
                "load_model",
                side_effect=lambda: events.append("model_loaded"),
            ) as preload,
            patch.object(analyzer, "analyze_file", return_value=expected),
        ):
            facade = PhonoMatch.__new__(PhonoMatch)
            facade._analyzer = analyzer
            facade._record = fake_recording
            result = facade.record_and_analyze_word(
                on_recording_complete=lambda: events.append("recording_complete"),
            )

        self.assertIs(result, expected)
        self.assertEqual(
            events,
            ["model_loaded", "recording_started", "recording_complete"],
        )
        preload.assert_called_once_with()

    def test_record_and_analyze_phrase_forwards_decoder_options(self) -> None:
        analyzer = Mock()
        expected = object()

        @contextmanager
        def fake_recording(_seconds: float) -> Generator[Path, None, None]:
            yield Path("recording.wav")

        with (
            patch.object(analyzer, "load_model"),
            patch.object(
                analyzer, "analyze_phrase_file", return_value=expected
            ) as analyze,
        ):
            facade = PhonoMatch.__new__(PhonoMatch)
            facade._analyzer = analyzer
            facade._record = fake_recording
            result = facade.record_and_analyze_phrase(
                seconds=3, beam_size=7, max_words=5
            )

        self.assertIs(result, expected)
        analyze.assert_called_once_with(Path("recording.wav"), beam_size=7, max_words=5)

    def test_convenience_wrapper_constructs_and_listens(self) -> None:
        expected = object()
        analyzer = Mock()
        analyzer.record_and_analyze_word.return_value = expected
        with (
            patch(
                "phonomatch.application.phonomatch.PhonoMatch",
                return_value=analyzer,
            ) as construct,
        ):
            result = listen_and_match(
                {"grun": "ɡrun"}, seconds=3, settings=MatchSettings()
            )
        self.assertIs(result, expected)
        construct.assert_called_once_with(
            {"grun": "ɡrun"}, MatchSettings(), default_words=False
        )
        analyzer.record_and_analyze_word.assert_called_once_with(seconds=3)
