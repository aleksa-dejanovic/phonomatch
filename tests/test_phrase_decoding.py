import math
import unittest
from typing import Any

import numpy as np

from phonomatch.infrastructure.phrase_decoding import align_tokens, decode_phrases


def _emissions(
    tokens: list[int], token_count: int = 3
) -> np.ndarray[Any, np.dtype[np.float64]]:
    probabilities = np.full((len(tokens), token_count), 0.005, dtype=np.float64)
    for frame, token in enumerate(tokens):
        probabilities[frame, token] = 0.99
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.log(probabilities)


class PhraseDecodingTests(unittest.TestCase):
    def test_decodes_multiple_vocabulary_words(self) -> None:
        hypotheses = decode_phrases(
            _emissions([0, 1, 0, 2, 0]),
            {"alpha": (1,), "beta": (2,)},
            blank_id=0,
        )

        self.assertEqual(hypotheses[0].words, ("alpha", "beta"))
        self.assertEqual(hypotheses[0].token_ids, (1, 2))

    def test_decodes_repeated_words_across_ctc_blank(self) -> None:
        hypotheses = decode_phrases(
            _emissions([1, 0, 1]),
            {"alpha": (1,)},
            blank_id=0,
        )

        self.assertEqual(hypotheses[0].words, ("alpha", "alpha"))

    def test_prefers_complete_lexicon_path(self) -> None:
        hypotheses = decode_phrases(
            _emissions([1, 0, 2]),
            {"short": (1,), "long": (1, 2)},
            blank_id=0,
        )

        self.assertEqual(hypotheses[0].words, ("long",))

    def test_returns_homophone_alternatives(self) -> None:
        hypotheses = decode_phrases(
            _emissions([1]),
            {"one": (1,), "won": (1,)},
            blank_id=0,
        )

        self.assertEqual(
            {hypothesis.words for hypothesis in hypotheses}, {("one",), ("won",)}
        )
        self.assertTrue(math.isclose(hypotheses[0].score, hypotheses[1].score))

    def test_aligns_output_tokens_to_frames(self) -> None:
        spans = align_tokens(_emissions([0, 1, 1, 0, 2, 2, 0]), (1, 2), blank_id=0)

        self.assertEqual(spans[0].start, 1)
        self.assertEqual(spans[0].end, 3)
        self.assertEqual(spans[1].start, 4)
        self.assertEqual(spans[1].end, 6)

    def test_rejects_invalid_decoder_arguments(self) -> None:
        with self.assertRaises(ValueError):
            decode_phrases(np.zeros((2, 3, 4)), {"word": (1,)}, blank_id=0)
        with self.assertRaises(ValueError):
            decode_phrases(_emissions([1]), {}, blank_id=0)
        with self.assertRaises(ValueError):
            align_tokens(np.empty((0, 3)), (1,), blank_id=0)


if __name__ == "__main__":
    unittest.main()
