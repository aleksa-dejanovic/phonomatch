import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from phonomatch import MatchSettings, PhonoMatch, listen_and_match
from phonomatch.infrastructure.recognition import (
    PhraseTranscription,
    RecognizedWord,
)


class AnalyzerTests(unittest.TestCase):
    def test_empty_vocabulary_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            PhonoMatch({})

    def test_match_ipa_returns_analysis_result(self) -> None:
        result = PhonoMatch({"grun": "ɡrun", "naku": "naku"}).match_ipa("gɾun")
        self.assertEqual(result.recognized_ipa, "gɾun")
        self.assertTrue(result.match.accepted)
        self.assertEqual(result.match.best_candidate.word, "grun")

    def test_analyze_file_forwards_model_revision(self) -> None:
        analyzer = PhonoMatch(
            {"grun": "ɡrun", "naku": "naku"},
            model_id="example/model",
            model_revision="immutable-commit",
        )
        with patch(
            "phonomatch.application.analyzer.speech_to_ipa", return_value="ɡrun"
        ) as recognize:
            analyzer.analyze_file("recording.wav")

        recognize.assert_called_once()
        self.assertEqual(recognize.call_args.kwargs["model_id"], "example/model")
        self.assertEqual(
            recognize.call_args.kwargs["model_revision"], "immutable-commit"
        )

    def test_load_model_forwards_configured_model(self) -> None:
        analyzer = PhonoMatch(
            {"grun": "ɡrun", "naku": "naku"},
            model_id="example/model",
            model_revision="immutable-commit",
        )
        with patch("phonomatch.application.analyzer.load_model") as preload:
            analyzer.load_model()

        preload.assert_called_once_with("example/model", "immutable-commit")

    def test_listen_reports_when_recording_is_complete(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun", "naku": "naku"})
        events: list[str] = []

        @contextmanager
        def fake_recording(_seconds: float) -> Iterator[Path]:
            events.append("recording_started")
            yield Path("recording.wav")

        expected = analyzer.match_ipa("gɾun")
        with (
            patch("phonomatch.application.analyzer.recorded_audio", fake_recording),
            patch.object(
                analyzer,
                "load_model",
                side_effect=lambda: events.append("model_loaded"),
            ) as preload,
            patch.object(analyzer, "analyze_file", return_value=expected),
        ):
            result = analyzer.listen(
                on_recording_complete=lambda: events.append("recording_complete")
            )

        self.assertIs(result, expected)
        self.assertEqual(
            events,
            ["model_loaded", "recording_started", "recording_complete"],
        )
        preload.assert_called_once_with()

    def test_analyze_phrase_file_matches_each_word_independently(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun", "naku": "naku"})
        transcription = PhraseTranscription(
            words=(
                RecognizedWord("grun", "ɡrun", 0.1, 0.6),
                RecognizedWord("naku", "naku", 0.7, 1.2),
            ),
            confidence=0.95,
            alternative=("grun", "grun"),
        )
        with patch(
            "phonomatch.application.analyzer.speech_to_phrase",
            return_value=transcription,
        ):
            result = analyzer.analyze_phrase_file("recording.wav")

        self.assertEqual(
            tuple(word.match.best_candidate.word for word in result.words),
            ("grun", "naku"),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.alternative_words, ("grun", "grun"))

    def test_ambiguous_phrase_sequence_is_rejected(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun"})
        transcription = PhraseTranscription(
            words=(RecognizedWord("grun", "ɡrun", 0.0, 0.5),),
            confidence=0.5,
            alternative=("grun", "grun"),
        )
        with patch(
            "phonomatch.application.analyzer.speech_to_phrase",
            return_value=transcription,
        ):
            result = analyzer.analyze_phrase_file("recording.wav")

        self.assertFalse(result.accepted)

    def test_words_returns_a_defensive_copy(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun"})
        words = dict(analyzer.words)
        words["extra"] = "extra"
        self.assertEqual(analyzer.words, {"grun": "ɡrun"})

    def test_listen_for_phrase_forwards_decoder_options(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun"})
        expected = object()

        @contextmanager
        def fake_recording(_seconds: float) -> Iterator[Path]:
            yield Path("recording.wav")

        with (
            patch("phonomatch.application.analyzer.recorded_audio", fake_recording),
            patch.object(analyzer, "load_model"),
            patch.object(
                analyzer, "analyze_phrase_file", return_value=expected
            ) as analyze,
        ):
            result = analyzer.listen_for_phrase(seconds=3, beam_size=7, max_words=5)

        self.assertIs(result, expected)
        analyze.assert_called_once_with(Path("recording.wav"), beam_size=7, max_words=5)

    def test_convenience_wrapper_constructs_and_listens(self) -> None:
        expected = object()
        with patch.object(PhonoMatch, "listen", return_value=expected) as listen:
            result = listen_and_match(
                {"grun": "ɡrun"}, seconds=3, settings=MatchSettings()
            )
        self.assertIs(result, expected)
        listen.assert_called_once_with(seconds=3)
