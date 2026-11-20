"""Causal hop loop over the stateless window API: units, speaker cuts and speakers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from .protocol import SAMPLE_RATE
from .speakers import CutConfig, PrimerBank, SpeakerCut, Word, score_with, speaker_cuts
from .speech import SpeechSegment
from .units import (
    SegmentDetector,
    SpeechUnit,
    UnitConfig,
    UnitJoiner,
    frame_energy_db,
    snap_edge,
)

Audio = npt.NDArray[np.float32]
WINDOW_SAMPLES = 8 * SAMPLE_RATE
Transcriber = Callable[[Audio, float], Sequence[Word]]


@dataclass(frozen=True)
class PipelineConfig:
    hop_s: float = 0.5
    units: UnitConfig = field(default_factory=UnitConfig)
    cuts: CutConfig = field(default_factory=CutConfig)
    max_speakers: int = 4
    max_primers: int = 3
    same_speaker_threshold: float = 0.5
    turn_end_threshold: float = 0.5


@dataclass(frozen=True)
class SpeakerSpan:
    start_s: float
    end_s: float
    speaker: int
    score: float | None


@dataclass
class PipelineResult:
    hops: list[tuple[float, float]] = field(default_factory=list)
    segments: list[SpeechSegment] = field(default_factory=list)
    raw_units: list[SpeechUnit] = field(default_factory=list)
    units: list[SpeechUnit] = field(default_factory=list)
    emitted_at_s: list[float] = field(default_factory=list)
    words: list[list[Word]] = field(default_factory=list)
    cuts: list[list[SpeakerCut]] = field(default_factory=list)
    spans: list[SpeakerSpan] = field(default_factory=list)


def window_ending_at(audio: Audio, end: int) -> Audio:
    """The eight seconds before ``end``; a shorter stream is padded on the left with zeros."""
    window = audio[max(0, end - WINDOW_SAMPLES) : end]
    return np.pad(window, (WINDOW_SAMPLES - len(window), 0))


def run_pipeline(
    model: Any,
    question: Any,
    audio: Audio,
    *,
    transcribe: Transcriber | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Replay ``audio`` hop by hop, reading only samples before the current hop end.

    ``question`` is the model's ``Question`` class. ``transcribe(audio, start_s)`` returns
    words on the stream timeline for one padded unit; without it units are not cut.
    """
    config = config or PipelineConfig()
    units_config = config.units
    speech = question.noul(
        "Is someone speaking?", seconds=1.0, threshold=units_config.threshold
    )
    turn_end = question.score("Has the speaker finished?")
    score_change = score_with(model, question.score("Did the speaker change?"))
    score_same = score_with(model, question.score("Is this the same speaker?"))
    detector, joiner = SegmentDetector(units_config), UnitJoiner(units_config)
    bank = PrimerBank(
        config.max_speakers, config.max_primers, config.same_speaker_threshold
    )
    result = PipelineResult()
    hop = round(config.hop_s * SAMPLE_RATE)
    margin = units_config.samples(units_config.margin_s)

    def probability(end: int, asked: Any) -> float:
        decision = model.decide_windows([window_ending_at(audio, end)], [asked])[0][0]
        return float(decision.probability)

    def join(segment: SpeechSegment) -> list[SpeechUnit]:
        result.segments.append(segment)
        ended = probability(segment.end, turn_end) >= config.turn_end_threshold
        return joiner.add(segment, ended)

    def emit(unit: SpeechUnit, now: int, following: int | None) -> None:
        heard = audio[:now]
        energy = frame_energy_db(heard, units_config)
        previous = result.units[-1] if result.units else None
        lower = previous.end if previous is not None else 0
        start = snap_edge(energy, unit.start, lower, unit.end, units_config)
        end = snap_edge(
            energy,
            unit.end,
            start,
            now if following is None else following,
            units_config,
        )
        # The earlier unit was padded before this one existed, so its padding also bounds.
        floor = (
            0
            if previous is None
            else max(previous.audio_end or 0, (previous.end + start + 1) // 2)
        )
        audio_start = max(floor, start - margin)
        audio_end = min(now, end + margin)
        if following is not None:
            audio_end = min(audio_end, max(end, (end + following) // 2))
        result.raw_units.append(unit)
        unit = replace(
            unit, start=start, end=end, audio_start=audio_start, audio_end=audio_end
        )
        result.units.append(unit)
        result.emitted_at_s.append(now / SAMPLE_RATE)
        words = (
            list(transcribe(heard[audio_start:audio_end], audio_start / SAMPLE_RATE))
            if transcribe is not None
            else []
        )

        def audio_at(start_s: float, end_s: float) -> Audio:
            return heard[
                max(0, round(start_s * SAMPLE_RATE)) : round(end_s * SAMPLE_RATE)
            ]

        cuts = speaker_cuts(words, score_change, audio_at, config.cuts) if words else []
        # Words come from the padded audio; a cut in the padding is outside the unit.
        cuts = [cut for cut in cuts if start < round(cut.time_s * SAMPLE_RATE) < end]
        result.words.append(words)
        result.cuts.append(cuts)
        edges = [start, *(round(cut.time_s * SAMPLE_RATE) for cut in cuts), end]
        for left, right in pairwise(edges):
            if right > left:
                speaker, score = bank.assign(heard[left:right], score_same)
                result.spans.append(
                    SpeakerSpan(left / SAMPLE_RATE, right / SAMPLE_RATE, speaker, score)
                )

    def emit_all(closed: list[SpeechUnit], now: int) -> None:
        for index, unit in enumerate(closed):
            later = closed[index + 1] if index + 1 < len(closed) else joiner.pending
            following = later.start if later is not None else detector.onset
            emit(unit, now, following)

    total = len(audio)
    for end in range(hop, total + 1, hop):
        value = probability(end, speech)
        result.hops.append((end / SAMPLE_RATE, value))
        closed: list[SpeechUnit] = []
        for segment in detector.observe(end / SAMPLE_RATE, value):
            closed += join(segment)
        closed += joiner.advance(end / SAMPLE_RATE, detector.onset)
        emit_all(closed, end)
    closed = []
    if detector.onset is not None and total - detector.onset >= units_config.samples(
        units_config.min_segment_s
    ):
        closed += join(SpeechSegment(detector.onset, total))
        detector.onset = None
    closed += joiner.finish()
    emit_all(closed, total)
    return result
