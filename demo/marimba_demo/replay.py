"""Run the demo scheduler, optionally waiting for every encoder hop."""

from __future__ import annotations

import asyncio
from math import ceil
from typing import Any

import numpy as np

from .protocol import HOP_SAMPLES, SAMPLE_RATE, Frame
from .scheduler import Scheduler
from .speech import OFFSET_LAG_S

HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE
TRAILING_HOPS = ceil((OFFSET_LAG_S + HOP_SECONDS) / HOP_SECONDS)
REPLAY_SILENCE_SECONDS = TRAILING_HOPS * HOP_SECONDS


def pcm16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


class SchedulerSession:
    def __init__(self, model: Any, questions: list[Any], send: Any) -> None:
        self.encoded = asyncio.Event()
        self.encoder_sequence = self.last_sequence = -1

        async def collect(payload: dict) -> None:
            await send(payload)
            # Turn results also use the encoder lane, but end at a speech offset.
            if (
                payload["type"] == "update"
                and payload["lane"] == "encoder"
                and payload["window_end_at_ms"]
                == (payload["sequence"] + 1) * HOP_SAMPLES / SAMPLE_RATE * 1000
            ):
                self.encoder_sequence = payload["sequence"]
                self.encoded.set()

        self.scheduler = Scheduler(model, questions, collect)
        if not self.scheduler.heads and self.scheduler.internal_speech_question is None:
            raise ValueError("An encoder question or speech capability is required.")
        self.workers = [
            asyncio.create_task(worker())
            for worker in (
                self.scheduler.live,
                self.scheduler.turn,
                self.scheduler.review,
            )
        ]

    def ingest(self, frame: Frame) -> None:
        self.scheduler.ingest(frame)
        self.last_sequence = frame.sequence

    async def wait_for_hop(self, sequence: int) -> None:
        while self.encoder_sequence < sequence:
            self.encoded.clear()
            waiter = asyncio.create_task(self.encoded.wait())
            try:
                done, _ = await asyncio.wait(
                    (waiter, self.workers[0]), return_when=asyncio.FIRST_COMPLETED
                )
                if self.workers[0] in done:
                    self.workers[0].result()
                    raise RuntimeError("Encoder stopped before emitting the hop.")
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    async def finish(self) -> None:
        # The final encoder pass may still enqueue turn and utterance work.
        await self.wait_for_hop(self.last_sequence)
        self.scheduler.finish()
        await asyncio.gather(*self.workers)

    async def abort(self) -> None:
        self.scheduler.finish(aborted=True)
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)


async def replay(model: Any, questions: list[Any], audio: np.ndarray) -> list[dict]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Replay requires nonempty finite mono audio at 16000 Hz.")
    events: list[dict] = []
    metadata = {
        "input_seconds": len(audio) / SAMPLE_RATE,
        "trailing_silence_s": REPLAY_SILENCE_SECONDS,
        "frame_padding_s": (-len(audio) % HOP_SAMPLES) / SAMPLE_RATE,
    }
    audio = np.pad(audio, (0, TRAILING_HOPS * HOP_SAMPLES))

    async def send(payload: dict) -> None:
        events.append({**payload, "replay": metadata})

    session = SchedulerSession(model, questions, send)
    try:
        for sequence, start in enumerate(range(0, len(audio), HOP_SAMPLES)):
            chunk = pcm16(audio[start : start + HOP_SAMPLES])
            chunk = np.pad(chunk, (0, HOP_SAMPLES - len(chunk)))
            session.ingest(Frame(sequence, sequence * 250.0, chunk))
            await session.wait_for_hop(sequence)
        await session.finish()
        return events
    finally:
        await session.abort()
