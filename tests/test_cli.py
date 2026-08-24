import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phonomatch.exceptions import RecognitionError
from phonomatch.interfaces.cli import (
    _percentage,
    _positive_float,
    _positive_int,
    _read_words,
    build_parser,
    main,
)


class CliTests(unittest.TestCase):
    def test_numeric_validators_reject_out_of_range_values(self) -> None:
        for validator, value in (
            (_positive_float, "0"),
            (_positive_float, "nan"),
            (_positive_float, "inf"),
            (_positive_int, "-1"),
            (_percentage, "2"),
        ):
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(value)

    def test_model_lifecycle_is_opt_in(self) -> None:
        self.assertFalse(build_parser().parse_args([]).enable_model_lifecycle)
        self.assertTrue(
            build_parser()
            .parse_args(["--enable-model-lifecycle"])
            .enable_model_lifecycle
        )

    def test_vocabulary_is_required_unless_default_words_are_requested(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error):
            status = main(["--color", "never"])
        self.assertEqual(status, 1)
        self.assertIn("use --words PATH or --default-words", error.getvalue())

        self.assertTrue(build_parser().parse_args(["--default-words"]).default_words)

    def test_reads_a_json_vocabulary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.json"
            path.write_text('{"grun": "ɡrun"}', encoding="utf-8")
            self.assertEqual(_read_words(path), {"grun": "ɡrun"})

    def test_rejects_unreadable_or_invalid_vocabulary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.json"
            with self.assertRaisesRegex(ValueError, "could not read vocabulary"):
                _read_words(path)

            path.write_text('["not a vocabulary"]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping words to IPA"):
                _read_words(path)

    def test_main_renders_accepted_word_result(self) -> None:
        analyzer = SimpleNamespace(
            load_model=lambda: None,
            listen=lambda **kwargs: SimpleNamespace(
                match=SimpleNamespace(accepted=True)
            ),
        )
        output = io.StringIO()
        with (
            patch("phonomatch.interfaces.cli.PhonoMatch", return_value=analyzer),
            patch(
                "phonomatch.interfaces.cli.render_result", return_value="word result"
            ),
            redirect_stdout(output),
        ):
            status = main(["--default-words", "--color", "never"])
        self.assertEqual(status, 0)
        self.assertIn("word result", output.getvalue())

    def test_main_returns_two_for_rejected_phrase(self) -> None:
        analyzer = SimpleNamespace(
            load_model=lambda: None,
            listen_for_phrase=lambda **kwargs: SimpleNamespace(accepted=False),
        )
        with (
            patch("phonomatch.interfaces.cli.PhonoMatch", return_value=analyzer),
            patch(
                "phonomatch.interfaces.cli.render_phrase_result",
                return_value="phrase result",
            ),
            redirect_stdout(io.StringIO()),
        ):
            status = main(["--default-words", "--phrase", "--color", "never"])
        self.assertEqual(status, 2)

    def test_main_renders_known_errors_to_stderr(self) -> None:
        error = io.StringIO()
        with (
            patch(
                "phonomatch.interfaces.cli.PhonoMatch",
                side_effect=RecognitionError("offline"),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(error),
        ):
            status = main(["--default-words", "--color", "never"])
        self.assertEqual(status, 1)
        self.assertIn("Error: offline", error.getvalue())
