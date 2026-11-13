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
from .speech import (
    TURN_OFFSET_LAG_S,
    SpeechSegment,
    SpeechTimeline,
    speaker_change_eligible,
)

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
        use_1s_turn_tail: bool = False,
    ) -> None:
        self.model, self.send = model, send
        self.speech = speech if speech is not None else SpeechTimeline()
        self.heads = [
            q for q in questions if model.lane(q) == "encoder" and q.task != "turn_end"
        ]
        self.turns = [q for q in questions if q.task == "turn_end"]
        self.reviews = [q for q in questions if model.lane(q) == "decoder"]
        self.term_reviews = [q for q in self.reviews if q.task == "term"]
        self.early_term_interval_s = getattr(model, "early_term_interval_s", None)
        if self.early_term_interval_s is not None and self.early_term_interval_s <= 0:
            raise ValueError("The early Term probe interval must be positive.")
        self.early_term_fired_questions: set[tuple[int, Any]] = set()
        self.early_term_busy = False
        self.internal_speech_question: Question | None = None
        self.internal_turn_speech_question: Question | None = None
        self.turn_speech_question: Question | None = None
        self.turn_speech: SpeechTimeline | None = None
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
        if use_1s_turn_tail and self.turns and self.speech_question is not None:
            self.turn_speech_question = next(
                (q for q in self.heads if self._is_turn_speech_question(q)), None
            )
            if self.turn_speech_question is None:
                self.internal_turn_speech_question = Question.noul(
                    _SPEECH_QUESTION_TEXT, task="voice_activity", seconds=1.0
                )
                self.turn_speech_question = self.internal_turn_speech_question
            self.turn_speech = SpeechTimeline(offset_lag_s=TURN_OFFSET_LAG_S)
        if len(self.heads) + len(self.turns) + len(self.reviews) != len(questions):
            raise ValueError("Unknown inference lane.")
        stream_questions = list(questions)
        for question in (
            self.internal_speech_question,
            self.internal_turn_speech_question,
        ):
            if question is not None and question not in stream_questions:
                stream_questions.append(question)
        self.stream = model.stream(
            stream_questions,
            window_seconds=8.0,
            context_seconds=8.0,
            hop_seconds=0.25,
            sample_rate=SAMPLE_RATE,
            coalesce=True,
        )
        self.changed = {
            lane: asyncio.Event()
            for lane in ("encoder", "turn", "decoder", "early_term")
        }
        self.turn_queue: deque[Work] = deque()
        self.utterance_queue: deque[list[Work]] = deque()
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

    @staticmethod
    def _is_turn_speech_question(question: Any) -> bool:
        return (
            question.task == "voice_activity"
            and question.text == _SPEECH_QUESTION_TEXT
            and question.kind == "noul"
            and getattr(question, "seconds", None) == 1.0
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

        if self.turn_speech_question is None:
            self.enqueue_turn(segment)
        if self.reviews:
            self.utterance_queue.append(
                [
                    self.make_work(s, e)
                    for s, e in utterance_chunks(segment.start, segment.end)
                ]
            )
            self.changed["decoder"].set()

    def make_work(self, start: int, end: int) -> Work:
        _, timestamp = self.latest_capture
        return Work(
            self.snapshot(start, end),
            self.latest_capture,
            timestamp + (end - self.latest_capture_start_sample) * 1000 / SAMPLE_RATE,
        )

    def enqueue_turn(self, segment: SpeechSegment) -> None:
        if not self.turns or segment.end <= segment.start:
            return
        self.turn_queue.append(
            self.make_work(max(0, segment.end - 8 * SAMPLE_RATE), segment.end)
        )
        self.changed["turn"].set()

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

    async def early_term(self) -> None:
        if self.early_term_interval_s is None or not self.term_reviews:
            return
        interval_samples = round(self.early_term_interval_s * SAMPLE_RATE)
        event = self.changed["early_term"]
        active_onset = None
        last_probe_end = None
        while not self.aborted and not self.finished:
            await event.wait()
            event.clear()
            if self.aborted or self.finished:
                return
            onset = self.speech.onset
            if onset is None:
                active_onset = None
                last_probe_end = None
                continue
            if active_onset != onset:
                active_onset = onset
                last_probe_end = onset
            probe_questions = [
                question
                for question in self.term_reviews
                if (onset, question) not in self.early_term_fired_questions
            ]
            if (
                not probe_questions
                or last_probe_end is None
                or self.total - last_probe_end < interval_samples
            ):
                continue

            end = self.total
            start = max(onset, end - 8 * SAMPLE_RATE)
            _, timestamp = self.latest_capture
            work = Work(
                self.snapshot(start, end),
                self.latest_capture,
                timestamp
                + (end - self.latest_capture_start_sample) * 1000 / SAMPLE_RATE,
            )
            last_probe_end = end
            self.early_term_busy = True
            try:
                decisions, elapsed = await asyncio.to_thread(
                    self.compute, work.snapshot, probe_questions, "decoder"
                )
            finally:
                self.early_term_busy = False
            if self.aborted or self.finished or self.speech.onset != onset:
                continue
            yes_decisions = [
                decision
                for decision in decisions
                if decision.task == "term" and decision.value is True
            ]
            if not yes_decisions:
                continue
            self.early_term_fired_questions.update(
                (onset, decision.question) for decision in yes_decisions
            )
            await self.send(
                {
                    "type": "update",
                    "lane": "decoder",
                    "sequence": work.capture[0],
                    "captured_at_ms": work.capture[1],
                    "window_end_at_ms": work.window_end_at_ms,
                    "decision_interval_ms": (
                        work.snapshot.window_end_s
                        - self.previous_end.get("decoder", 0.0)
                    )
                    * 1000,
                    "coalesced_hops": self.coalesced["decoder"],
                    "queued_utterances": len(self.utterance_queue),
                    "decisions": [
                        {
                            **decision.to_dict(),
                            "window_latency_ms": elapsed,
                        }
                        for decision in yes_decisions
                    ],
                }
            )

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
        if lane == "encoder" and self.turn_speech_question is not None:
            turn_speech_decision = next(
                (
                    decision
                    for decision in decisions
                    if decision.question == self.turn_speech_question
                ),
                None,
            )
            if (
                turn_speech_decision is not None
                and isinstance(turn_speech_decision.value, bool)
                and self.turn_speech is not None
            ):
                for segment in self.turn_speech.observe(
                    snapshot.window_end_s, turn_speech_decision.value
                ):
                    self.enqueue_turn(segment)
        sequence, timestamp = work.capture
        visible_decisions = [
            d
            for d in decisions
            if d.question != self.internal_speech_question
            and d.question != self.internal_turn_speech_question
            and d.question not in neutral
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
                "queued_utterances": len(self.utterance_queue),
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
                for question in (
                    self.internal_speech_question,
                    self.internal_turn_speech_question,
                ):
                    if question is not None and question not in questions:
                        questions.append(question)
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
            while self.utterance_queue and not self.aborted:
                utterance = self.utterance_queue.popleft()
                tasks = [q.task for q in self.reviews]
                await self.send(
                    {"type": "review_status", "tasks": tasks, "running": True}
                )
                groups = (
                    [q for q in self.reviews if q.task == "language"],
                    [q for q in self.reviews if q.task == "term"],
                    [q for q in self.reviews if q.task not in {"language", "term"}],
                )

                async def review_group(
                    utterance: list[Work], questions: list[Any], *, skip: bool = False
                ) -> None:
                    if not questions:
                        return
                    if skip:
                        if not self.aborted:
                            await self.send(
                                {
                                    "type": "review_status",
                                    "tasks": list(
                                        dict.fromkeys(q.task for q in questions)
                                    ),
                                    "running": False,
                                }
                            )
                        return
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

                utterance_start = round(
                    utterance[0].snapshot.window_start_s * SAMPLE_RATE
                )
                term_group = groups[1]
                term_questions = [
                    question
                    for question in term_group
                    if (utterance_start, question)
                    not in self.early_term_fired_questions
                ]
                term_already_probed = bool(term_group) and not term_questions
                self.early_term_fired_questions.difference_update(
                    (utterance_start, question) for question in term_group
                )
                term_review_questions = term_questions if term_questions else term_group
                if getattr(self.model, "concurrent_decoder", False):
                    await asyncio.gather(
                        review_group(utterance, groups[0]),
                        review_group(
                            utterance,
                            term_review_questions,
                            skip=term_already_probed,
                        ),
                    )
                    if self.aborted:
                        return
                    await review_group(utterance, groups[2])
                else:
                    review_groups = (
                        groups[0],
                        term_review_questions,
                        groups[2],
                    )
                    for index, questions in enumerate(review_groups):
                        await review_group(
                            utterance,
                            questions,
                            skip=index == 1 and term_already_probed,
                        )
                        if self.aborted:
                            return
            if self.finished:
                return
