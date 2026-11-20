"""Word-level speaker cuts and a causal primer bank, on the public pair questions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .protocol import SAMPLE_RATE

Audio = npt.NDArray[np.float32]
SENTENCE_END = (".", "?", "!", "\u2026")


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class CutConfig:
    threshold: float = 0.5
    min_peak_gap_s: float = 2.0
    sentence_snap_s: float = 1.5
    candidate_step_s: float = 0.5
    reference_s: float = 6.0
    segment_s: float = 2.0


@dataclass(frozen=True)
class SpeakerCut:
    """A cut after ``words[word_index]``; ``peak_s`` is where the change score peaked."""

    word_index: int
    time_s: float
    peak_s: float
    probability: float
    at_sentence_end: bool


def cut_candidates(words: Sequence[Word], config: CutConfig | None = None) -> list[int]:
    """Indexes of words whose end is scored, at most one per ``candidate_step_s``."""
    config = config or CutConfig()
    chosen: list[int] = []
    for index, word in enumerate(words[:-1]):
        if (
            not chosen
            or word.end_s - words[chosen[-1]].end_s >= config.candidate_step_s
        ):
            chosen.append(index)
    return chosen


def change_peaks(
    scores: Sequence[tuple[float, float]], config: CutConfig | None = None
) -> list[tuple[float, float]]:
    """Scores at or above the threshold, strongest first, at least the gap apart in time."""
    config = config or CutConfig()
    kept: list[tuple[float, float]] = []
    for time_s, probability in sorted(scores, key=lambda item: (-item[1], item[0])):
        if probability < config.threshold:
            break
        if all(abs(time_s - other) >= config.min_peak_gap_s for other, _ in kept):
            kept.append((time_s, probability))
    return sorted(kept)


def snap_cuts(
    words: Sequence[Word],
    peaks: Sequence[tuple[float, float]],
    config: CutConfig | None = None,
) -> list[SpeakerCut]:
    """Place each peak at the nearest sentence end in reach, else the nearest word end."""
    config = config or CutConfig()
    ends = np.array([word.end_s for word in words[:-1]])
    if not len(ends):
        return []
    sentence = np.array(
        [word.text.rstrip("\"')]").endswith(SENTENCE_END) for word in words[:-1]]
    )
    cuts: dict[int, SpeakerCut] = {}
    for peak_s, probability in peaks:
        distance = np.abs(ends - peak_s)
        reachable = sentence & (distance <= config.sentence_snap_s)
        index = int(
            np.argmin(
                np.where(reachable, distance, np.inf) if reachable.any() else distance
            )
        )
        cut = SpeakerCut(
            index, float(ends[index]), peak_s, probability, bool(reachable.any())
        )
        if index not in cuts or cuts[index].probability < probability:
            cuts[index] = cut
    return [cuts[index] for index in sorted(cuts)]


def speaker_cuts(
    words: Sequence[Word],
    score_changes: Callable[[list[tuple[Audio, Audio]]], Sequence[float]],
    audio_at: Callable[[float, float], Audio],
    config: CutConfig | None = None,
) -> list[SpeakerCut]:
    """Score ``speaker_change`` at word ends, keep separated peaks, snap them to sentences.

    ``audio_at(start_s, end_s)`` returns stream audio, clipped to what is retained.
    ``score_changes`` answers one batch of ``(reference, segment)`` pairs, for example
    ``lambda pairs: [d.probability for d in model.decide_pairs(question, pairs)]``.
    """
    config = config or CutConfig()
    candidates = cut_candidates(words, config)
    pairs = []
    scored = []
    for index in candidates:
        time_s = words[index].end_s
        reference = audio_at(time_s - config.reference_s, time_s)
        segment = audio_at(time_s, min(time_s + config.segment_s, words[-1].end_s))
        if len(reference) and len(segment):
            pairs.append((reference, segment))
            scored.append(time_s)
    if not pairs:
        return []
    probabilities = score_changes(pairs)
    return snap_cuts(
        words,
        change_peaks(list(zip(scored, probabilities, strict=True)), config),
        config,
    )


@dataclass
class PrimerBank:
    """Speaker references learned in stream order; nothing is known before the first span."""

    max_speakers: int = 4
    max_primers: int = 3
    threshold: float = 0.5
    min_primer_s: float = 2.0
    primer_s: float = 4.0
    primers: list[list[Audio]] = field(default_factory=list)

    def assign(
        self,
        audio: Audio,
        score_same: Callable[[list[tuple[Audio, Audio]]], Sequence[float]],
    ) -> tuple[int, float | None]:
        """Return the speaker index for ``audio`` and the score that chose it."""
        owners = [index for index, group in enumerate(self.primers) for _ in group]
        pairs = [(primer, audio) for group in self.primers for primer in group]
        best: int | None = None
        best_score: float | None = None
        if pairs:
            scores = np.asarray(score_same(pairs), dtype=np.float64)
            means = [
                float(scores[[owner == index for owner in owners]].mean())
                for index in range(len(self.primers))
            ]
            best = int(np.argmax(means))
            best_score = means[best]
        matched = (
            best is not None and best_score is not None and best_score >= self.threshold
        )
        created = not matched and len(self.primers) < self.max_speakers
        if created:
            self.primers.append([])
            best = len(self.primers) - 1
        assert best is not None
        long_enough = len(audio) >= round(self.min_primer_s * SAMPLE_RATE)
        # A new speaker needs one reference whatever its length. A span forced onto its
        # closest speaker because the bank is full is not trusted as a primer.
        if created or (
            matched and long_enough and len(self.primers[best]) < self.max_primers
        ):
            self.primers[best].append(
                audio[: round(self.primer_s * SAMPLE_RATE)].copy()
            )
        return best, best_score


def score_with(model: Any, question: Any, **options: Any) -> Callable[..., list[float]]:
    """Adapt ``model.decide_pairs`` to the scoring callables above."""

    def score(pairs: list[tuple[Audio, Audio]]) -> list[float]:
        return [
            float(d.probability) for d in model.decide_pairs(question, pairs, **options)
        ]

    return score
