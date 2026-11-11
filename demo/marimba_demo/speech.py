"""CPU speech endpoints and coverage on the original stream timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .protocol import SAMPLE_RATE


@dataclass(frozen=True)
class SpeechSegment:
    start: int
    end: int

    @property
    def seconds(self) -> tuple[float, float]:
        return self.start / SAMPLE_RATE, self.end / SAMPLE_RATE


class SpeechTimeline:
    def __init__(self, iterator: Any = None) -> None:
        if iterator is None:
            import torch

            threads = torch.get_num_threads()
            try:
                from silero_vad import VADIterator, load_silero_vad
            finally:
                # Silero's import must not change Marimba's CPU thread budget.
                torch.set_num_threads(threads)

            iterator = VADIterator(
                load_silero_vad(onnx=True),
                threshold=0.5,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=100,
                speech_pad_ms=0,
            )
        self.iterator = iterator
        self.pending: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.segments: list[SpeechSegment] = []
        self.onset: int | None = None
        self.processed = 0

    def ingest(self, pcm: np.ndarray) -> list[SpeechSegment]:
        data = pcm.astype(np.float32) / 32768.0
        self.pending = np.concatenate((self.pending, data))
        closed = []
        while len(self.pending) >= 512:
            event = self.iterator(self.pending[:512], return_seconds=False)
            self.pending = self.pending[512:]
            self.processed += 512
            if event and "start" in event:
                self.onset = int(event["start"])
            if event and "end" in event and self.onset is not None:
                segment = SpeechSegment(self.onset, int(event["end"]))
                self.segments.append(segment)
                closed.append(segment)
                self.onset = None
        return closed

    def coverage(self, start_s: float, end_s: float) -> float:
        start, end = start_s * SAMPLE_RATE, end_s * SAMPLE_RATE
        samples = sum(
            max(0, min(end, s.end) - max(start, s.start)) for s in self.segments
        )
        if self.onset is not None:
            samples += max(0, min(end, self.processed) - max(start, self.onset))
        return samples / SAMPLE_RATE


def speaker_change_eligible(snapshot: Any, speech: SpeechTimeline) -> bool:
    start = snapshot.window_start_s
    return bool(
        snapshot.mask.all()
        and speech.coverage(start, start + 6) >= 0.5
        and speech.coverage(start + 6, snapshot.window_end_s) >= 0.5
    )
