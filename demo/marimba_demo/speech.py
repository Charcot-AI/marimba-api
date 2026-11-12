"""Speech endpoints and coverage on the original stream timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import SAMPLE_RATE

# These lags match the measured flip delays of the two-second readout.
ONSET_LAG_S = 0.5
OFFSET_LAG_S = 1.75


@dataclass(frozen=True)
class SpeechSegment:
    start: int
    end: int

    @property
    def seconds(self) -> tuple[float, float]:
        return self.start / SAMPLE_RATE, self.end / SAMPLE_RATE


class SpeechTimeline:
    def __init__(self) -> None:
        self.segments: list[SpeechSegment] = []
        self.onset: int | None = None
        self.last_observed_end = 0

    def observe(self, window_end_s: float, speaking: bool) -> list[SpeechSegment]:
        window_end = round(window_end_s * SAMPLE_RATE)
        self.last_observed_end = max(self.last_observed_end, window_end)
        if speaking:
            if self.onset is None:
                self.onset = max(0, window_end - round(ONSET_LAG_S * SAMPLE_RATE))
            return []
        if self.onset is None:
            return []
        segment = SpeechSegment(
            self.onset,
            max(self.onset, window_end - round(OFFSET_LAG_S * SAMPLE_RATE)),
        )
        self.segments.append(segment)
        self.onset = None
        return [segment]

    def coverage(self, start_s: float, end_s: float) -> float:
        start, end = round(start_s * SAMPLE_RATE), round(end_s * SAMPLE_RATE)
        samples = sum(
            max(0, min(end, s.end) - max(start, s.start)) for s in self.segments
        )
        if self.onset is not None:
            samples += max(0, min(end, self.last_observed_end) - max(start, self.onset))
        return samples / SAMPLE_RATE


def speaker_change_eligible(snapshot: Any, speech: SpeechTimeline) -> bool:
    start = snapshot.window_start_s
    return bool(
        snapshot.mask.all()
        and speech.coverage(start, start + 6) >= 0.5
        and speech.coverage(start + 6, snapshot.window_end_s) >= 0.5
    )
