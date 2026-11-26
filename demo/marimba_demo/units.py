"""Speech units for transcription, built causally from the 1 s voice-activity readout."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .protocol import SAMPLE_RATE
from .speech import ONSET_LAG_S, TURN_OFFSET_LAG_S, SpeechSegment


@dataclass(frozen=True)
class UnitConfig:
    threshold: float = 0.1
    onset_lag_s: float = ONSET_LAG_S
    offset_lag_s: float = TURN_OFFSET_LAG_S
    min_segment_s: float = 0.25
    max_pause_s: float = 5.0
    max_unit_s: float = 30.0
    snap_search_s: float = 0.5
    energy_frame_s: float = 0.01
    energy_smooth_frames: int = 5
    margin_s: float = 0.25

    def samples(self, seconds: float) -> int:
        return round(seconds * SAMPLE_RATE)


@dataclass(frozen=True)
class SpeechUnit:
    """A joined run of segments; ``start`` and ``end`` are samples on the stream timeline."""

    start: int
    end: int
    turn_ended: bool = False
    audio_start: int | None = None
    audio_end: int | None = None

    @property
    def seconds(self) -> tuple[float, float]:
        return self.start / SAMPLE_RATE, self.end / SAMPLE_RATE


class SegmentDetector:
    """Open at ``p >= threshold``, close on the first hop below it; no hysteresis."""

    def __init__(self, config: UnitConfig | None = None) -> None:
        self.config = config or UnitConfig()
        self.onset: int | None = None

    def observe(self, window_end_s: float, probability: float) -> list[SpeechSegment]:
        config = self.config
        window_end = config.samples(window_end_s)
        if probability >= config.threshold:
            if self.onset is None:
                self.onset = max(0, window_end - config.samples(config.onset_lag_s))
                return []
            # A long run is sliced while still open so a unit never waits past its cap.
            limit = config.samples(config.max_unit_s)
            if window_end - config.samples(config.offset_lag_s) - self.onset >= limit:
                segment = SpeechSegment(self.onset, self.onset + limit)
                self.onset = segment.end
                return [segment]
            return []
        if self.onset is None:
            return []
        segment = SpeechSegment(
            self.onset, window_end - config.samples(config.offset_lag_s)
        )
        self.onset = None
        if segment.end - segment.start < config.samples(config.min_segment_s):
            return []
        return [segment]


class UnitJoiner:
    """Join segments while no turn ended, the pause is short and the unit fits its cap."""

    def __init__(self, config: UnitConfig | None = None) -> None:
        self.config = config or UnitConfig()
        self.pending: SpeechUnit | None = None

    def add(self, segment: SpeechSegment, turn_ended: bool = False) -> list[SpeechUnit]:
        config = self.config
        limit = config.samples(config.max_unit_s)
        closed: list[SpeechUnit] = []
        pieces = [
            (start, min(start + limit, segment.end))
            for start in range(segment.start, segment.end, limit)
        ]
        for index, (start, end) in enumerate(pieces):
            last = index == len(pieces) - 1
            pending = self.pending
            if (
                pending is not None
                and not pending.turn_ended
                and start - pending.end <= config.samples(config.max_pause_s)
                and end - pending.start <= limit
            ):
                self.pending = replace(pending, end=end, turn_ended=turn_ended and last)
                continue
            if pending is not None:
                closed.append(pending)
            self.pending = SpeechUnit(start, end, turn_ended and last)
        pending = self.pending
        # Nothing can join a unit that ended a turn or already fills the cap.
        if pending is not None and (
            pending.turn_ended or pending.end - pending.start >= limit
        ):
            closed.append(pending)
            self.pending = None
        return closed

    def advance(
        self, window_end_s: float, onset: int | None = None
    ) -> list[SpeechUnit]:
        """Close the pending unit once no later segment can start within the pause.

        ``onset`` is the start of a segment that is still open, if any.
        """
        config = self.config
        pending = self.pending
        if pending is None:
            return []
        earliest = (
            onset
            if onset is not None
            else config.samples(window_end_s) - config.samples(config.onset_lag_s)
        )
        if earliest - pending.end > config.samples(config.max_pause_s):
            self.pending = None
            return [pending]
        return []

    def finish(self) -> list[SpeechUnit]:
        pending, self.pending = self.pending, None
        return [] if pending is None else [pending]


def frame_energy_db(
    audio: npt.NDArray[np.float32], config: UnitConfig | None = None
) -> npt.NDArray[np.float64]:
    """Mean-square energy per frame, centered moving average, in dB."""
    config = config or UnitConfig()
    size = config.samples(config.energy_frame_s)
    count = len(audio) // size
    if count == 0:
        return np.empty(0, dtype=np.float64)
    frames = audio[: count * size].astype(np.float64).reshape(count, size)
    energy = np.mean(frames**2, axis=1)
    width = config.energy_smooth_frames
    kernel = np.ones(width)
    # Dividing by the number of frames present keeps the edges from looking quieter.
    smoothed = np.convolve(energy, kernel, "same") / np.convolve(
        np.ones(count), kernel, "same"
    )
    return np.asarray(10 * np.log10(smoothed + 1e-12))


def snap_edge(
    energy_db: npt.NDArray[np.float64],
    edge: int,
    lower: int,
    upper: int,
    config: UnitConfig | None = None,
    *,
    energy_start: int = 0,
) -> int:
    """Move ``edge`` to the quietest frame boundary within the search range.

    ``energy_db`` covers the stream from sample ``energy_start``. ``lower`` and ``upper``
    are the neighboring unit edges (or the audio bounds) the result must stay within.
    """
    config = config or UnitConfig()
    size = config.samples(config.energy_frame_s)
    search = config.samples(config.snap_search_s)
    first = max(lower, edge - search, energy_start)
    last = min(upper, edge + search, energy_start + len(energy_db) * size)
    frames = np.arange(
        -(-(first - energy_start) // size), (last - energy_start) // size
    )
    if not len(frames):
        return edge
    level = np.round(energy_db[frames])
    centers = energy_start + frames * size + size // 2
    quietest = frames[level == level.min()]
    best = quietest[np.argmin(np.abs(centers[level == level.min()] - edge))]
    return int(energy_start + best * size + size // 2)


def snap_units(
    audio: npt.NDArray[np.float32],
    units: list[SpeechUnit],
    config: UnitConfig | None = None,
    *,
    audio_start: int = 0,
) -> list[SpeechUnit]:
    """Snap both edges of each unit to the energy minimum, then pad for transcription.

    ``audio`` holds the stream from sample ``audio_start``; unit edges are stream samples.
    """
    config = config or UnitConfig()
    energy = frame_energy_db(audio, config)
    audio_end = audio_start + len(audio)
    snapped: list[SpeechUnit] = []
    for index, unit in enumerate(units):
        lower = snapped[-1].end if snapped else audio_start
        upper = units[index + 1].start if index + 1 < len(units) else audio_end
        start = snap_edge(
            energy, unit.start, lower, unit.end, config, energy_start=audio_start
        )
        end = snap_edge(
            energy, unit.end, start, upper, config, energy_start=audio_start
        )
        snapped.append(replace(unit, start=start, end=end))
    margin = config.samples(config.margin_s)
    padded = []
    for index, unit in enumerate(snapped):
        before = snapped[index - 1].end if index else None
        after = snapped[index + 1].start if index + 1 < len(snapped) else None
        padded.append(
            replace(
                unit,
                audio_start=max(
                    audio_start,
                    unit.start - margin,
                    audio_start if before is None else (before + unit.start + 1) // 2,
                ),
                audio_end=min(
                    audio_end,
                    unit.end + margin,
                    audio_end if after is None else (unit.end + after) // 2,
                ),
            )
        )
    return padded
