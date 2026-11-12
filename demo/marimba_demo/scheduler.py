"""Rolling heads, offset-triggered turns, and bounded utterance decoding."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
from marimba import Question
from marimba.audio import right_aligned_window
from marimba.windows import WindowSnapshot

from .protocol import SAMPLE_RATE, Frame
from .speech import SpeechSegment, SpeechTimeline, speaker_change_eligible

_SPEECH_QUESTION_TEXT = "Is someone speaking?"


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
        self.internal_speech_question: Question | None = None
        self.speech_question: Any | None = next(
            (q for q in self.heads if self._is_speech_question(q)), None
        )
        capabilities = getattr(model, "capabilities", {})
        voice_activity = capabilities.get("voice_activity")
        if self.speech_question is None and (
            any(q.task == "voice_activity" for q in self.heads)
            or (voice_activity is not None and voice_activity.status != "unavailable")
        ):
            self.internal_speech_question = Question.noul(
                _SPEECH_QUESTION_TEXT, task="voice_activity", seconds=2.0
            )
            self.speech_question = self.internal_speech_question
        if len(self.heads) + len(self.turns) + len(self.reviews) != len(questions):
            raise ValueError("Unknown inference lane.")
        stream_questions = list(questions)
        if self.internal_speech_question is not None:
            stream_questions.append(self.internal_speech_question)
        self.stream = model.stream(
            stream_questions,
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
        self.latest_capture_start_sample = 0
        self.finished = self.aborted = False
        self.previous_end: dict[str, float] = {}
        self.coalesced = {lane: 0 for lane in self.changed}

    @staticmethod
    def _is_speech_question(question: Any) -> bool:
        return (
            question.task == "voice_activity"
            and question.text == _SPEECH_QUESTION_TEXT
            and question.kind == "noul"
            and getattr(question, "seconds", None) == 2.0
            and getattr(question, "threshold", None) is None
            and getattr(question, "options", None) is None
        )

    def snapshot(self, start: int, end: int) -> WindowSnapshot:
        if start < self.audio_start or end > self.total:
            raise ValueError(
                f"Snapshot range [{start}, {end}) is outside retained audio "
                f"[{self.audio_start}, {self.total})"
            )
        audio = self.audio[start - self.audio_start : end - self.audio_start]
        window, mask = right_aligned_window(audio)
        return WindowSnapshot(window, mask, start / SAMPLE_RATE, end / SAMPLE_RATE)

    def close_utterance(self, segment: SpeechSegment) -> None:
        if segment.end <= segment.start:
            return

        def work(start: int, end: int) -> Work:
            _, timestamp = self.latest_capture
            return Work(
                self.snapshot(start, end),
                self.latest_capture,
                timestamp
                + (end - self.latest_capture_start_sample) * 1000 / SAMPLE_RATE,
            )

        if self.turns:
            self.turn_queue.append(
                work(max(0, segment.end - 8 * SAMPLE_RATE), segment.end)
            )
            self.changed["turn"].set()
        if self.reviews:
            if self.pending_utterance is not None:
                self.dropped_utterances += 1
            self.pending_utterance = [
                work(s, e) for s, e in utterance_chunks(segment.start, segment.end)
            ]
            self.changed["decoder"].set()

    def ingest(self, frame: Frame) -> None:
        self.latest_capture_start_sample = self.total
        self.audio = np.concatenate(
            (self.audio, frame.pcm.astype(np.float32) / 32768.0)
        )
        self.total += len(frame.pcm)
        self.latest_capture = (frame.sequence, frame.captured_at_ms)
        self.stream.ingest(frame.pcm)
        retention_samples = 10 * SAMPLE_RATE + SAMPLE_RATE // 2
        keep = min(
            self.total - retention_samples,
            self.speech.last_observed_end - retention_samples,
            self.speech.onset if self.speech.onset is not None else self.total,
        )
        keep = max(0, keep)
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
        speaking_onset_s: float | None = None
        interval = snapshot.window_end_s - self.previous_end.get(lane, 0.0)
        if lane == "encoder":
            self.coalesced[lane] += max(0, round(interval / 0.25) - 1)
        self.previous_end[lane] = snapshot.window_end_s
        if lane == "encoder" and self.speech_question is not None:
            speech_decision = next(
                (
                    decision
                    for decision in decisions
                    if decision.question == self.speech_question
                ),
                None,
            )
            if speech_decision is not None and isinstance(speech_decision.value, bool):
                previous_onset = self.speech.onset
                for segment in self.speech.observe(
                    snapshot.window_end_s, speech_decision.value
                ):
                    self.close_utterance(segment)
                if (
                    previous_onset is None
                    and self.speech.onset is not None
                    and self.turns
                ):
                    speaking_onset_s = self.speech.onset / SAMPLE_RATE
        sequence, timestamp = work.capture
        visible_decisions = [
            d
            for d in decisions
            if d.question != self.internal_speech_question and d.question not in neutral
        ]
        payloads = []
        for decision in visible_decisions:
            payload = {
                **decision.to_dict(),
                "window_latency_ms": elapsed,
            }
            payloads.append(payload)
        if speaking_onset_s is not None:
            question = self.turns[0]
            payloads.append(
                {
                    "task": "turn_end",
                    "question": asdict(question),
                    "kind": question.kind,
                    "value": False,
                    "probability": None,
                    "probabilities": None,
                    "score": None,
                    "threshold": None,
                    "status": "speaking",
                    "window_start_s": speaking_onset_s,
                    "window_end_s": speaking_onset_s,
                    "latency_ms": 0,
                    "window_latency_ms": 0,
                    "calibrated": False,
                }
            )
        for question in neutral:
            payload = {
                "task": question.task,
                "question": question.text,
                "kind": question.kind,
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
            payloads.append(payload)
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
        if not self.heads and self.internal_speech_question is None:
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
                questions = [
                    q
                    for q in self.heads
                    if q.task != "speaker_count" or snapshot.mask.all()
                ]
                neutral = [
                    q
                    for q in questions
                    if q.task == "speaker_change"
                    and not speaker_change_eligible(snapshot, self.speech)
                ]
                questions = [q for q in questions if q not in neutral]
                if (
                    self.internal_speech_question is not None
                    and self.internal_speech_question not in questions
                ):
                    questions.append(self.internal_speech_question)
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
                groups = (
                    [q for q in self.reviews if q.task == "language"],
                    [q for q in self.reviews if q.task == "term"],
                    [q for q in self.reviews if q.task not in {"language", "term"}],
                )
                for questions in groups:
                    if not questions:
                        continue
                    for work in utterance:
                        if self.aborted:
                            return
                        await self.emit(work, questions, "decoder")
                    if not self.aborted:
                        await self.send(
                            {
                                "type": "review_status",
                                "tasks": list(dict.fromkeys(q.task for q in questions)),
                                "running": False,
                            }
                        )
            if self.finished:
                return
