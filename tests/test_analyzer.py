import unittest
from unittest.mock import patch

from phonomatch import DEFAULT_WORDS, PhonoMatch
from phonomatch.infrastructure.recognition import (
    PhraseTranscription,
    RecognizedWord,
)


class AnalyzerTests(unittest.TestCase):
    def test_vocabulary_is_required_unless_defaults_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "word list is required"):
            PhonoMatch()

        self.assertEqual(PhonoMatch(default_words=True).words, DEFAULT_WORDS)

    def test_explicit_and_default_vocabularies_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "either words or default_words"):
            PhonoMatch({"grun": "ɡrun"}, default_words=True)

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

    def test_unload_model_releases_cached_models(self) -> None:
        analyzer = PhonoMatch({"grun": "ɡrun"})
        with patch("phonomatch.application.analyzer.unload_models") as unload:
            analyzer.unload_model()

        unload.assert_called_once_with()

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
