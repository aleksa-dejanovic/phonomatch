"""Lexicon-constrained CTC decoding and alignment for spoken phrases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class PhraseHypothesis:
    """A vocabulary word sequence decoded from CTC emissions."""

    words: tuple[str, ...]
    token_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class TokenSpan:
    """Inclusive-exclusive frame span assigned to one output token."""

    start: int
    end: int


@dataclass
class _BeamEntry:
    blank_score: float = -math.inf
    nonblank_score: float = -math.inf

    @property
    def score(self) -> float:
        return _log_add(self.blank_score, self.nonblank_score)


@dataclass
class _TrieNode:
    children: dict[int, _TrieNode]
    words: list[str]

    def __init__(self) -> None:
        self.children = {}
        self.words = []


BeamState = tuple[tuple[str, ...], tuple[int, ...]]


def decode_phrases(
    log_probabilities: np.ndarray[Any, np.dtype[np.float64]],
    pronunciations: Mapping[str, Sequence[int]],
    blank_id: int,
    *,
    beam_size: int = 64,
    nbest: int = 2,
    max_words: int = 12,
) -> tuple[PhraseHypothesis, ...]:
    """Decode the strongest vocabulary-constrained CTC word sequences."""
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape (frames, tokens)")
    if not 0 <= blank_id < log_probabilities.shape[1]:
        raise ValueError("blank_id is outside the token vocabulary")
    if beam_size < 1 or nbest < 1 or max_words < 1:
        raise ValueError("beam_size, nbest, and max_words must be positive")

    root = _build_trie(pronunciations)
    nodes = _index_nodes(root)
    beams: dict[BeamState, _BeamEntry] = {((), ()): _BeamEntry(blank_score=0.0)}

    for frame_value in log_probabilities:
        frame = cast(np.ndarray[Any, np.dtype[np.float64]], frame_value)
        next_beams: dict[BeamState, _BeamEntry] = {}
        for state, entry in beams.items():
            completed, current = state
            total = entry.score
            target = next_beams.setdefault(state, _BeamEntry())
            target.blank_score = _log_add(
                target.blank_score, total + float(frame[blank_id])
            )

            last_token = _last_token(completed, current, pronunciations)
            for token, next_state in _extensions(
                completed, current, root, nodes, max_words
            ):
                token_score = float(frame[token])
                extended = next_beams.setdefault(next_state, _BeamEntry())
                if token == last_token:
                    extended.nonblank_score = _log_add(
                        extended.nonblank_score,
                        entry.blank_score + token_score,
                    )
                    target.nonblank_score = _log_add(
                        target.nonblank_score,
                        entry.nonblank_score + token_score,
                    )
                else:
                    extended.nonblank_score = _log_add(
                        extended.nonblank_score, total + token_score
                    )

        beams = dict(
            sorted(next_beams.items(), key=lambda item: item[1].score, reverse=True)[
                :beam_size
            ]
        )

    completed_hypotheses: list[PhraseHypothesis] = []
    for (words, current), entry in beams.items():
        node = nodes.get(current)
        if node is None or not node.words:
            continue
        token_ids = (
            tuple(token for word in words for token in pronunciations[word]) + current
        )
        for word in node.words:
            completed_hypotheses.append(
                PhraseHypothesis((*words, word), token_ids, entry.score)
            )

    completed_hypotheses.sort(key=lambda hypothesis: hypothesis.score, reverse=True)
    return tuple(completed_hypotheses[:nbest])


def align_tokens(
    log_probabilities: np.ndarray[Any, np.dtype[np.float64]],
    token_ids: Sequence[int],
    blank_id: int,
) -> tuple[TokenSpan, ...]:
    """Viterbi-align a known CTC token sequence to emission frames."""
    if not token_ids:
        return ()
    frame_count, token_count = log_probabilities.shape
    if frame_count == 0:
        raise ValueError("at least one emission frame is required")
    labels = [blank_id]
    for token in token_ids:
        if not 0 <= token < token_count:
            raise ValueError("token ID is outside the emission vocabulary")
        labels.extend((token, blank_id))

    state_count = len(labels)
    scores = np.full((frame_count, state_count), -math.inf, dtype=np.float64)
    parents = np.full((frame_count, state_count), -1, dtype=np.int32)
    scores[0, 0] = float(log_probabilities[0, blank_id])
    scores[0, 1] = float(log_probabilities[0, labels[1]])

    for frame_index in range(1, frame_count):
        for state_index, label in enumerate(labels):
            choices = [(scores[frame_index - 1, state_index], state_index)]
            if state_index > 0:
                choices.append(
                    (scores[frame_index - 1, state_index - 1], state_index - 1)
                )
            if (
                state_index > 1
                and label != blank_id
                and label != labels[state_index - 2]
            ):
                choices.append(
                    (scores[frame_index - 1, state_index - 2], state_index - 2)
                )
            previous_score, previous_state = max(choices)
            scores[frame_index, state_index] = previous_score + float(
                log_probabilities[frame_index, label]
            )
            parents[frame_index, state_index] = previous_state

    final_state = max((state_count - 2, state_count - 1), key=lambda i: scores[-1, i])
    if not math.isfinite(float(scores[-1, final_state])):
        raise ValueError("the token sequence cannot be aligned to the available frames")

    path = [final_state]
    for frame_index in range(frame_count - 1, 0, -1):
        path.append(int(parents[frame_index, path[-1]]))
    path.reverse()

    spans: list[TokenSpan] = []
    for token_index in range(len(token_ids)):
        state_index = 2 * token_index + 1
        frames = [index for index, state in enumerate(path) if state == state_index]
        if not frames:
            raise ValueError("a decoded token has no aligned frames")
        spans.append(TokenSpan(frames[0], frames[-1] + 1))
    return tuple(spans)


def _build_trie(pronunciations: Mapping[str, Sequence[int]]) -> _TrieNode:
    if not pronunciations:
        raise ValueError("at least one pronunciation is required")
    root = _TrieNode()
    for word, tokens in pronunciations.items():
        if not tokens:
            raise ValueError(f"pronunciation for {word!r} contains no tokens")
        node = root
        for token in tokens:
            node = node.children.setdefault(int(token), _TrieNode())
        node.words.append(word)
    return root


def _index_nodes(root: _TrieNode) -> dict[tuple[int, ...], _TrieNode]:
    nodes: dict[tuple[int, ...], _TrieNode] = {}

    def visit(node: _TrieNode, prefix: tuple[int, ...]) -> None:
        nodes[prefix] = node
        for token, child in node.children.items():
            visit(child, (*prefix, token))

    visit(root, ())
    return nodes


def _extensions(
    completed: tuple[str, ...],
    current: tuple[int, ...],
    root: _TrieNode,
    nodes: Mapping[tuple[int, ...], _TrieNode],
    max_words: int,
) -> list[tuple[int, BeamState]]:
    node = nodes[current]
    extensions = [(token, (completed, (*current, token))) for token in node.children]
    if node.words and len(completed) + 1 < max_words:
        for word in node.words:
            for token in root.children:
                extensions.append((token, ((*completed, word), (token,))))
    return extensions


def _last_token(
    completed: tuple[str, ...],
    current: tuple[int, ...],
    pronunciations: Mapping[str, Sequence[int]],
) -> int | None:
    if current:
        return current[-1]
    if completed:
        return int(pronunciations[completed[-1]][-1])
    return None


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))
