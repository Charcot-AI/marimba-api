"""Independent live and buffered-review workers over one continuous audio ring."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from time import perf_counter
from typing import Any

from .protocol import SAMPLE_RATE, Frame


def cadence_settings(values: dict[str, float] | None) -> dict[str, float]:
    result = dict(values or {})
    if any(not math.isfinite(v) or v < 0.25 for v in result.values()):
        raise ValueError("Review intervals must be finite and at least 0.25 seconds.")
    return result


def review_context(value: float | str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 8 <= seconds <= 30:
        raise ValueError("Review context must be between 8 and 30 seconds.")
    return seconds


class Scheduler:
    def __init__(
        self,
        model: Any,
        questions: list[Any],
        send: Any,
        *,
        review_seconds: float = 8.0,
        intervals: dict[str, float] | None = None,
    ) -> None:
        review_seconds = review_context(review_seconds)
        self.model, self.send, self.review_seconds = model, send, review_seconds
        self.intervals = cadence_settings(intervals)
        self.heads = [q for q in questions if model.lane(q) == "encoder"]
        self.reviews = [q for q in questions if model.lane(q) == "decoder"]
        if len(self.heads) + len(self.reviews) != len(questions):
            raise ValueError("Unknown inference lane.")
        self.stream = model.stream(
            questions,
            window_seconds=8.0,
            context_seconds=review_seconds,
            hop_seconds=0.25,
            sample_rate=SAMPLE_RATE,
            coalesce=True,
        )
        self.changed = {"encoder": asyncio.Event(), "decoder": asyncio.Event()}
        self.captures: OrderedDict[float, tuple[int, float]] = OrderedDict()
        self.frames = 0
        self.finished = self.aborted = False
        self.previous_end: dict[str, float] = {}
        self.coalesced = {"encoder": 0, "decoder": 0}

    def ingest(self, frame: Frame) -> None:
        self.frames += 1
        self.captures[self.frames * 0.25] = (frame.sequence, frame.captured_at_ms)
        if len(self.captures) > 128:
            self.captures.popitem(last=False)
        self.stream.ingest(frame.pcm)
        for event in self.changed.values():
            event.set()

    def finish(self, *, aborted: bool = False) -> None:
        self.finished = True
        self.aborted |= aborted
        for event in self.changed.values():
            event.set()

    def capture(self, snapshot: Any) -> tuple[int, float]:
        return self.captures[round(snapshot.window_end_s, 6)]

    def compute(
        self, snapshot: Any, questions: list[Any], lane: str
    ) -> tuple[list[Any], float]:
        started = perf_counter()
        try:
            if lane == "encoder":
                encoded = self.model.encode(snapshot)
                decisions = self.model.decide_encoded(encoded, questions)
            else:
                decisions = self.model.decide_snapshot(snapshot, questions)
        except Exception as error:
            raise RuntimeError("Model inference failed.") from error
        return decisions, (perf_counter() - started) * 1000

    async def emit(
        self, snapshot: Any, capture: tuple[int, float], questions: list[Any], lane: str
    ) -> None:
        decisions, elapsed = await asyncio.to_thread(
            self.compute, snapshot, questions, lane
        )
        if self.aborted:
            return
        previous = self.previous_end.get(lane, 0.0)
        interval = snapshot.window_end_s - previous
        self.coalesced[lane] += max(0, round(interval / 0.25) - 1)
        self.previous_end[lane] = snapshot.window_end_s
        sequence, timestamp = capture
        await self.send(
            {
                "type": "update",
                "lane": lane,
                "sequence": sequence,
                "captured_at_ms": timestamp,
                "decision_interval_ms": interval * 1000,
                "coalesced_hops": self.coalesced[lane],
                "decisions": [
                    {
                        **d.to_dict(),
                        "window_latency_ms": elapsed,
                        **(
                            {"status": "unvalidated"}
                            if lane == "decoder" and self.review_seconds > 8
                            else {}
                        ),
                    }
                    for d in decisions
                ],
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
                capture = self.capture(snapshot)
                previous = snapshot.window_end_s
                await self.emit(snapshot, capture, self.heads, "encoder")
            if self.finished and not event.is_set():
                return

    async def review(self) -> None:
        if not self.reviews:
            return
        due = {q.task: 0.0 for q in self.reviews}
        previous = {q.task: 0.0 for q in self.reviews}
        event = self.changed["decoder"]
        while not self.finished:
            event.clear()
            delay = None
            snapshot = self.stream.snapshot(seconds=self.review_seconds)
            if snapshot is not None and snapshot.window_end_s >= self.review_seconds:
                eligible = [
                    q for q in self.reviews if snapshot.window_end_s > previous[q.task]
                ]
                now = asyncio.get_running_loop().time()
                questions = [q for q in eligible if due[q.task] <= now]
                if questions:
                    capture = self.capture(snapshot)
                    await self.send(
                        {
                            "type": "review_status",
                            "tasks": [q.task for q in questions],
                            "running": True,
                        }
                    )
                    await self.emit(snapshot, capture, questions, "decoder")
                    completed = asyncio.get_running_loop().time()
                    for q in questions:
                        previous[q.task] = snapshot.window_end_s
                        due[q.task] = completed + self.intervals.get(q.task, 2.0)
                    if not self.aborted:
                        await self.send(
                            {
                                "type": "review_status",
                                "tasks": [q.task for q in questions],
                                "running": False,
                            }
                        )
                    continue
                if eligible:
                    delay = max(0.0, min(due[q.task] for q in eligible) - now)
            if self.finished:
                return
            try:
                await asyncio.wait_for(event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
