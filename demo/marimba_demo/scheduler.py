"""Rolling heads, offset-triggered turns, and bounded utterance decoding."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
from marimba.audio import right_aligned_window
from marimba.windows import WindowSnapshot

from .protocol import SAMPLE_RATE, Frame
from .speech import SpeechSegment, SpeechTimeline, speaker_change_eligible


def utterance_chunks(start: int, end: int) -> list[tuple[int, int]]:
    return [
        (s, min(s + 8 * SAMPLE_RATE, end)) for s in range(start, end, 8 * SAMPLE_RATE)
    ]


@dataclass(frozen=True)
class Work:
    snapshot: Any
    capture: tuple[int, float]
    window_end_at_ms: float | None = None


class Scheduler:
    def __init__(
        self,
        model: Any,
        questions: list[Any],
        send: Any,
        *,
        speech: SpeechTimeline | None = None,
    ) -> None:
        self.model, self.send = model, send
        self.speech = speech if speech is not None else SpeechTimeline()
        self.heads = [
            q for q in questions if model.lane(q) == "encoder" and q.task != "turn_end"
        ]
        self.turns = [q for q in questions if q.task == "turn_end"]
        self.reviews = [q for q in questions if model.lane(q) == "decoder"]
        if len(self.heads) + len(self.turns) + len(self.reviews) != len(questions):
            raise ValueError("Unknown inference lane.")
        self.stream = model.stream(
            questions,
            window_seconds=8.0,
            context_seconds=8.0,
            hop_seconds=0.25,
            sample_rate=SAMPLE_RATE,
            coalesce=True,
        )
        self.changed = {
            lane: asyncio.Event() for lane in ("encoder", "turn", "decoder")
        }
        self.turn_queue: deque[Work] = deque()
        self.pending_utterance: list[Work] | None = None
        self.dropped_utterances = 0
        self.audio: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.audio_start = self.total = 0
        self.latest_capture = (0, 0.0)
        self.finished = self.aborted = False
        self.previous_end: dict[str, float] = {}
        self.coalesced = {lane: 0 for lane in self.changed}

    def snapshot(self, start: int, end: int) -> WindowSnapshot:
        audio = self.audio[start - self.audio_start : end - self.audio_start]
        window, mask = right_aligned_window(audio)
        return WindowSnapshot(window, mask, start / SAMPLE_RATE, end / SAMPLE_RATE)

    def close_utterance(self, segment: SpeechSegment, frame: Frame) -> None:
        def work(start: int, end: int) -> Work:
            # Capture timestamps refer to the first sample of a browser packet.
            timestamp = (
                frame.captured_at_ms + (end - (self.total - len(frame.pcm))) / 16
            )
            return Work(
                self.snapshot(start, end),
                (frame.sequence, frame.captured_at_ms),
                timestamp,
            )

        if self.turns:
            self.turn_queue.append(
                work(max(0, segment.end - 8 * SAMPLE_RATE), segment.end)
            )
        if self.reviews:
            if self.pending_utterance is not None:
                self.dropped_utterances += 1
            self.pending_utterance = [
                work(s, e) for s, e in utterance_chunks(segment.start, segment.end)
            ]

    def ingest(self, frame: Frame) -> None:
        self.audio = np.concatenate(
            (self.audio, frame.pcm.astype(np.float32) / 32768.0)
        )
        self.total += len(frame.pcm)
        self.latest_capture = (frame.sequence, frame.captured_at_ms)
        self.stream.ingest(frame.pcm)
        for segment in self.speech.ingest(frame.pcm):
            self.close_utterance(segment, frame)
        keep = max(0, self.total - 9 * SAMPLE_RATE)
        if self.speech.onset is not None:
            keep = min(keep, self.speech.onset)
        self.audio = self.audio[keep - self.audio_start :]
        self.audio_start = keep
        for event in self.changed.values():
            event.set()

    def finish(self, *, aborted: bool = False) -> None:
        self.finished = True
        self.aborted |= aborted
        for event in self.changed.values():
            event.set()

    def compute(
        self, snapshot: Any, questions: list[Any], lane: str
    ) -> tuple[list[Any], float]:
        started = perf_counter()
        try:
            if not questions:
                decisions = []
            elif lane != "decoder":
                decisions = self.model.decide_encoded(
                    self.model.encode(snapshot), questions
                )
            else:
                decisions = self.model.decide_snapshot(snapshot, questions)
        except Exception as error:
            raise RuntimeError("Model inference failed.") from error
        return decisions, (perf_counter() - started) * 1000

    async def emit(
        self,
        work: Work,
        questions: list[Any],
        lane: str,
        neutral: tuple[Any, ...] | list[Any] = (),
    ) -> None:
        snapshot = work.snapshot
        decisions, elapsed = await asyncio.to_thread(
            self.compute, snapshot, questions, lane
        )
        if self.aborted:
            return
        interval = snapshot.window_end_s - self.previous_end.get(lane, 0.0)
        if lane == "encoder":
            self.coalesced[lane] += max(0, round(interval / 0.25) - 1)
        self.previous_end[lane] = snapshot.window_end_s
        sequence, timestamp = work.capture
        payloads = [{**d.to_dict(), "window_latency_ms": elapsed} for d in decisions]
        payloads.extend(
            {
                "task": q.task,
                "question": q.text,
                "kind": q.kind,
                "value": None,
                "probability": None,
                "probabilities": None,
                "score": None,
                "threshold": None,
                "status": "not_enough_speech",
                "window_start_s": snapshot.window_start_s,
                "window_end_s": snapshot.window_end_s,
                "latency_ms": 0,
                "window_latency_ms": 0,
            }
            for q in neutral
        )
        await self.send(
            {
                "type": "update",
                "lane": "decoder" if lane == "decoder" else "encoder",
                "sequence": sequence,
                "captured_at_ms": timestamp,
                "window_end_at_ms": work.window_end_at_ms
                if work.window_end_at_ms is not None
                else timestamp + 250,
                "decision_interval_ms": interval * 1000,
                "coalesced_hops": self.coalesced[lane],
                "dropped_utterances": self.dropped_utterances,
                "decisions": payloads,
            }
        )

    async def live(self) -> None:
        if not self.heads:
            return
        previous = 0.0
        event = self.changed["encoder"]
        while not self.aborted:
            await event.wait()
            event.clear()
            if self.aborted:
                return
            snapshot = self.stream.snapshot()
            if snapshot is not None and snapshot.window_end_s > previous:
                previous = snapshot.window_end_s
                neutral = [
                    q
                    for q in self.heads
                    if q.task == "speaker_change"
                    and not speaker_change_eligible(snapshot, self.speech)
                ]
                questions = [
                    q
                    for q in self.heads
                    if q not in neutral
                    and (q.task != "speaker_count" or snapshot.mask.all())
                ]
                await self.emit(
                    Work(snapshot, self.latest_capture), questions, "encoder", neutral
                )
            if self.finished and not event.is_set():
                return

    async def turn(self) -> None:
        if not self.turns:
            return
        event = self.changed["turn"]
        while not self.aborted:
            await event.wait()
            event.clear()
            while self.turn_queue and not self.aborted:
                await self.emit(self.turn_queue.popleft(), self.turns, "turn")
            if self.finished:
                return

    async def review(self) -> None:
        if not self.reviews:
            return
        event = self.changed["decoder"]
        while not self.aborted:
            await event.wait()
            event.clear()
            while self.pending_utterance is not None and not self.aborted:
                utterance, self.pending_utterance = self.pending_utterance, None
                tasks = [q.task for q in self.reviews]
                await self.send(
                    {"type": "review_status", "tasks": tasks, "running": True}
                )
                for work in utterance:
                    if self.aborted:
                        return
                    await self.emit(work, self.reviews, "decoder")
                if not self.aborted:
                    await self.send(
                        {"type": "review_status", "tasks": tasks, "running": False}
                    )
            if self.finished:
                return
