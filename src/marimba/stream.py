"""Incremental ring-buffer streaming; each sample is ingested once."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .audio import AudioArray, Mask, normalize_array, right_aligned_window
from .decision import Decision, StreamUpdate
from .questions import Question
from .runtime import RuntimeInfo, synchronize


class WindowModel(Protocol):
    runtime: RuntimeInfo

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]: ...


class Stream:
    def __init__(
        self,
        model: WindowModel,
        questions: list[Question],
        window_seconds: float = 8.0,
        hop_seconds: float = 0.25,
        sample_rate: int = 16000,
        coalesce: bool = False,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("Stream chunks must already be sampled at 16000 Hz")
        if not np.isfinite([window_seconds, hop_seconds]).all():
            raise ValueError("Window and hop must be finite")
        self.size = round(window_seconds * sample_rate)
        self.hop = round(hop_seconds * sample_rate)
        if min(self.size, self.hop) <= 0:
            raise ValueError("Window and hop must be positive")
        self.model, self.questions, self.sample_rate = model, list(questions), sample_rate
        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.position = self.total = self.pending = 0
        self.coalesce = coalesce
        self._lock = RLock()

    def _ingest(self, chunk: AudioArray) -> None:
        count = len(chunk)
        if count >= self.size:
            chunk = chunk[-self.size :]
            self.buffer[:] = chunk
            self.position = 0
        else:
            first = min(count, self.size - self.position)
            self.buffer[self.position : self.position + first] = chunk[:first]
            self.buffer[: count - first] = chunk[first:]
            self.position = (self.position + count) % self.size
        self.total += count
        self.pending += count

    def ingest(self, chunk: Any) -> None:
        data = normalize_array(chunk)
        with self._lock:
            self._ingest(data)

    def _snapshot(self) -> tuple[AudioArray, Mask, float, float]:
        count = min(self.total, self.size)
        ordered = np.concatenate((self.buffer[self.position :], self.buffer[: self.position]))
        window, mask = right_aligned_window(ordered[-count:] if count else ordered[:0], self.size)
        end = self.total / self.sample_rate
        start = max(0.0, (self.total - self.size) / self.sample_rate)
        self.pending = 0
        return window, mask, start, end

    def _score(self, snapshot: tuple[AudioArray, Mask, float, float]) -> StreamUpdate:
        window, mask, start, end = snapshot
        synchronize(self.model.runtime)
        started = perf_counter()
        decisions = self.model.decide_window(window, mask, self.questions, start, end)
        synchronize(self.model.runtime)
        latency = (perf_counter() - started) * 1000
        return StreamUpdate(
            start, end, [replace(d, latency_ms=latency) for d in decisions], latency
        )

    def decide(self) -> StreamUpdate | None:
        with self._lock:
            if not self.total:
                return None
            snapshot = self._snapshot()
        return self._score(snapshot)

    def push(self, chunk: Any) -> list[StreamUpdate]:
        data = normalize_array(chunk)
        if not len(data):
            return []
        if self.coalesce:
            with self._lock:
                self.pending %= self.hop
                due = (self.pending + len(data)) // self.hop
                if not due:
                    self._ingest(data)
                    return []
                count = due * self.hop - self.pending
                self._ingest(data[:count])
                latest_snapshot = self._snapshot()
                self._ingest(data[count:])
            return [self._score(latest_snapshot)]
        updates = []
        offset = 0
        while offset < len(data):
            with self._lock:
                self.pending %= self.hop
                count = min(self.hop - self.pending, len(data) - offset)
                self._ingest(data[offset : offset + count])
                offset += count
                snapshot = self._snapshot() if self.pending == self.hop else None
            if snapshot is not None:
                updates.append(self._score(snapshot))
        return updates

    def flush(self) -> list[StreamUpdate]:
        with self._lock:
            if not self.pending:
                return []
            snapshot = self._snapshot()
        return [self._score(snapshot)]
