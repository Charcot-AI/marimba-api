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
from .windows import MAX_DECODER_SECONDS, WindowSnapshot


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
        context_seconds: float = 8.0,
        decoder_hop_seconds: float | None = None,
        decoder_context_seconds: float | None = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("Stream chunks must already be sampled at 16000 Hz")
        if not np.isfinite([window_seconds, hop_seconds]).all():
            raise ValueError("Window and hop must be finite")
        if not np.isfinite(context_seconds) or not 0 < context_seconds <= MAX_DECODER_SECONDS:
            raise ValueError("Context must be positive and at most MAX_DECODER_SECONDS")
        self.window_size = round(window_seconds * sample_rate)
        self.size = max(self.window_size, round(context_seconds * sample_rate))
        self.hop = round(hop_seconds * sample_rate)
        if min(self.window_size, self.hop) <= 0:
            raise ValueError("Window and hop must be positive")
        self.model, self.questions, self.sample_rate = model, list(questions), sample_rate
        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.position = self.total = self.pending = 0
        self.coalesce = coalesce
        self.decoder_hop_seconds = (
            hop_seconds
            if decoder_hop_seconds is None and decoder_context_seconds is not None
            else decoder_hop_seconds
        )
        self.decoder_context_seconds = (
            context_seconds if decoder_context_seconds is None else decoder_context_seconds
        )
        if (
            not np.isfinite(self.decoder_context_seconds)
            or not 0 < self.decoder_context_seconds <= self.size / sample_rate
            or self.decoder_context_seconds > MAX_DECODER_SECONDS
        ):
            raise ValueError("Decoder context must fit the retained ring and MAX_DECODER_SECONDS")
        if decoder_hop_seconds is not None and (
            not np.isfinite(decoder_hop_seconds) or decoder_hop_seconds <= 0
        ):
            raise ValueError("Decoder hop must be finite and positive")
        self._last_decoder_end: float | None = None
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

    def _snapshot(self, seconds: float | None = None, reset: bool = True) -> WindowSnapshot:
        size = self.window_size if seconds is None else round(seconds * self.sample_rate)
        if size <= 0 or size > self.size:
            raise ValueError("Snapshot duration must fit the retained context")
        count = min(self.total, size)
        ordered = np.concatenate((self.buffer[self.position :], self.buffer[: self.position]))
        window, mask = right_aligned_window(ordered[-count:] if count else ordered[:0], size)
        end = self.total / self.sample_rate
        start = max(0.0, (self.total - size) / self.sample_rate)
        if reset:
            self.pending = 0
        return WindowSnapshot(window, mask, start, end)

    def snapshot(self, seconds: float | None = None) -> WindowSnapshot | None:
        if seconds is not None and not np.isfinite(seconds):
            raise ValueError("Snapshot duration must be finite")
        with self._lock:
            if seconds is not None and not 0 < seconds <= self.size / self.sample_rate:
                raise ValueError("Snapshot duration must fit the retained context")
            return self._snapshot(seconds, reset=False) if self.total else None

    def _prepare(self) -> tuple[WindowSnapshot, WindowSnapshot | None]:
        snapshot = self._snapshot()
        decoder = None
        if self.decoder_hop_seconds is not None and (
            self._last_decoder_end is None
            or snapshot.window_end_s - self._last_decoder_end >= self.decoder_hop_seconds - 1e-9
        ):
            decoder = self._snapshot(self.decoder_context_seconds, reset=False)
            self._last_decoder_end = snapshot.window_end_s
        return snapshot, decoder

    def _score(
        self,
        snapshots: tuple[WindowSnapshot, WindowSnapshot | None],
        questions: list[Question] | None = None,
    ) -> StreamUpdate:
        snapshot, decoder = snapshots
        start, end = snapshot.window_start_s, snapshot.window_end_s
        selected = self.questions if questions is None else questions
        synchronize(self.model.runtime)
        started = perf_counter()
        if self.decoder_hop_seconds is None or questions is not None:
            if (
                questions is not None
                and selected
                and snapshot.seconds == 8
                and hasattr(self.model, "encode")
            ):
                model: Any = self.model
                decisions = model.decide_encoded(model.encode(snapshot), selected)
            else:
                decisions = self.model.decide_window(
                    snapshot.audio, snapshot.mask, selected, start, end
                )
        else:
            model = self.model
            encoder_questions = [q for q in selected if model.lane(q) == "encoder"]
            decoder_questions = (
                [q for q in selected if model.lane(q) == "decoder"] if decoder else []
            )
            if decoder is not None and decoder.seconds == snapshot.seconds:
                computed = model.decide_encoded(
                    model.encode(snapshot), encoder_questions + decoder_questions
                )
            else:
                computed = (
                    model.decide_encoded(model.encode(snapshot), encoder_questions)
                    if encoder_questions
                    else []
                )
                if decoder_questions:
                    computed += model.decide_snapshot(decoder, decoder_questions)
            by_question = {id(d.question): d for d in computed}
            decisions = [by_question[id(q)] for q in selected if id(q) in by_question]
        synchronize(self.model.runtime)
        latency = (perf_counter() - started) * 1000
        return StreamUpdate(
            start, end, [replace(d, latency_ms=latency) for d in decisions], latency
        )

    def decide(self, questions: list[Question] | None = None) -> StreamUpdate | None:
        with self._lock:
            if not self.total:
                return None
            snapshot = self._snapshot()
        return self._score((snapshot, None), self.questions if questions is None else questions)

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
                latest_snapshot = self._prepare()
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
                snapshot = self._prepare() if self.pending == self.hop else None
            if snapshot is not None:
                updates.append(self._score(snapshot))
        return updates

    def flush(self) -> list[StreamUpdate]:
        with self._lock:
            if not self.pending:
                return []
            snapshot = self._prepare()
        return [self._score(snapshot)]
