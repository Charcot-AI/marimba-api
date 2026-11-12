"""Per-session PCM buffering and lifecycle for the unchanged demo scheduler."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import numpy as np
import soxr
from marimba_demo.protocol import HOP_SAMPLES, SAMPLE_RATE, Frame
from marimba_demo.replay import SchedulerSession, pcm16

from marimba.audio import normalize_array
from space.presentation import UTTERANCE


class LiveSession:
    def __init__(self, model: Any, questions: list[Any], term: str, session_id: str) -> None:
        self.loop = asyncio.get_running_loop()
        self.term, self.session_id = term, session_id
        self.latest: dict[str, dict] = {}
        self.utterances: deque[dict] = deque(maxlen=64)
        self.buffer = np.empty(0, np.float32)
        self.rate: int | None = None
        self.resampler: Any = None
        self.sequence = 0
        self.closed = False
        self.closing = asyncio.Lock()
        self.run = SchedulerSession(model, questions, self.send)

    async def send(self, payload: dict) -> None:
        for decision in payload.get("decisions", []):
            self.latest[decision["task"]] = decision
            if decision["task"] in UTTERANCE:
                self.utterances.append(decision)

    def feed(self, audio: np.ndarray) -> None:
        self.buffer = np.concatenate((self.buffer, audio))
        while len(self.buffer) >= HOP_SAMPLES:
            self.run.ingest(
                Frame(self.sequence, self.sequence * 250.0, pcm16(self.buffer[:HOP_SAMPLES]))
            )
            self.sequence += 1
            self.buffer = self.buffer[HOP_SAMPLES:]

    def push(self, chunk: tuple[int, np.ndarray]) -> None:
        for worker in self.run.workers:
            if worker.done():
                worker.result()
        rate, array = chunk
        if rate <= 0 or (self.rate is not None and self.rate != rate):
            raise ValueError("The microphone sample rate changed. Stop and restart recording.")
        if self.rate is None:
            self.rate = rate
            if rate != SAMPLE_RATE:
                self.resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
        audio = normalize_array(array)
        if self.resampler is not None:
            audio = self.resampler.resample_chunk(audio)
        self.feed(audio)

    async def close(self, *, aborted: bool = False) -> None:
        async with self.closing:
            await self._close(aborted=aborted)

    async def _close(self, *, aborted: bool) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if not aborted:
                if self.resampler is not None:
                    self.feed(self.resampler.resample_chunk(np.empty(0, np.float32), last=True))
                await self.run.finish()
        finally:
            await self.run.abort()


class LiveController:
    def __init__(self, model: Any, make_questions: Any) -> None:
        self.model, self.make_questions = model, make_questions
        self.owner: LiveSession | None = None

    def start(self, state: LiveSession | None, term: str, session_id: str) -> LiveSession:
        if self.owner is not None:
            if self.owner.closed or self.owner.session_id != session_id:
                raise ValueError("Busy: another live session is active. Try again after it stops.")
            state = self.owner
        if state is None or state.closed:
            state = LiveSession(
                self.model, self.make_questions(self.model, term), term.strip(), session_id
            )
            self.owner = state
        return state

    async def stream(
        self, chunk: tuple[int, np.ndarray], state: LiveSession | None, term: str, session_id: str
    ) -> LiveSession:
        state = self.start(state, term, session_id)
        try:
            if state.term != term.strip():
                raise ValueError("Stop recording before changing the term.")
            state.push(chunk)
        except Exception:
            await self.close(state, aborted=True)
            raise
        return state

    async def close(self, state: LiveSession | None, *, aborted: bool = False) -> None:
        if state is not None:
            try:
                await state.close(aborted=aborted)
            finally:
                if self.owner is state:
                    self.owner = None

    async def disconnect(self, session_id: str) -> None:
        if self.owner is not None and self.owner.session_id == session_id:
            await self.close(self.owner, aborted=True)

    def delete_state(self, state: LiveSession | None) -> None:
        if state is not None and not state.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.close(state, aborted=True), state.loop)
