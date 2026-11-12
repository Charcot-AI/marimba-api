"""Vapi custom transcriber: energy timing, Marimba decisions, independent STT."""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from time import perf_counter

import numpy as np
import soxr

from marimba import Marimba, Question

VOICE = Question.noul("Is someone speaking?", task="voice_activity")
TURN = Question.score("Has the speaker finished their turn?", task="turn_end")
LANGUAGE = Question.choice("What language is being spoken?", ("en", "pt", "es"), task="language")
RATE, HOP, WINDOW = 16000, 320, 128000
logger = logging.getLogger("vapi_bridge")


def log_event(event):
    logger.info(json.dumps(event))


def whisper_stt() -> Callable[[np.ndarray], str]:
    from transformers import pipeline

    transcribe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    return lambda audio: transcribe({"raw": audio, "sampling_rate": RATE})["text"].strip()


class Bridge:
    def __init__(
        self,
        model,
        stt,
        *,
        threshold=0.5,
        partial_seconds=1.0,
        silence_timeout=2.5,
        max_turn_seconds=30.0,
        emit=None,
    ):
        if not all(
            np.isfinite(x) for x in (threshold, partial_seconds, silence_timeout, max_turn_seconds)
        ):
            raise ValueError("Nonfinite settings")
        if (
            not 0 <= threshold <= 1
            or partial_seconds < 1
            or silence_timeout < 0.3
            or max_turn_seconds < 1
        ):
            raise ValueError("Invalid timing settings")
        self.model, self.stt = model, stt
        self.threshold, self.partial_seconds = threshold, partial_seconds
        self.timeout = round(silence_timeout * RATE)
        self.max_turn = round(max_turn_seconds * RATE)
        self.emit = emit or log_event
        self.audio = np.empty(0, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.base = self.total = self.offset = self.voiced = self.onset = 0
        self.start = None
        self.run = 0
        self.floor = 0.001
        self.scored = None
        self.last_partial = 0.0
        self.partial_at = 0
        self.final_at = 0.0
        self.finalizing = False
        self.language_job = None
        self.turn_count = 0
        self.captured_at = perf_counter()

    def ingest(self, audio, captured_at=None):
        self.pending = np.concatenate((self.pending, audio))
        self.captured_at = perf_counter() if captured_at is None else captured_at
        count = len(self.pending) // HOP * HOP
        for frame in self.pending[:count].reshape(-1, HOP):
            self.audio = np.concatenate((self.audio, frame))
            self.total += HOP
            energy = float(np.sqrt(np.mean(frame * frame)))
            speaking = energy >= max(0.006, self.floor * 3)
            if speaking:
                if not self.run:
                    self.onset = self.total - HOP
                self.run += HOP
                if self.start is None and self.run >= 3 * HOP:
                    self.start = max(self.base, self.onset - 3200)
                    self.voiced = self.run
                elif self.start is not None:
                    self.voiced += HOP
                self.offset = self.total
                self.scored = None
            else:
                self.run = 0
                self.floor = 0.98 * self.floor + 0.02 * min(energy, 0.01)
            keep = (
                min(self.total - WINDOW, self.start)
                if self.start is not None
                else self.total - WINDOW
            )
            discard = max(0, keep - self.base)
            self.audio = self.audio[discard:]
            self.base += discard
            if len(self.audio) > self.max_turn + WINDOW:
                raise BufferError("Inference capacity exceeded")
        self.pending = self.pending[count:]

    def slice(self, start, end):
        return self.audio[max(start, self.base) - self.base : end - self.base].copy()

    def window(self, end):
        audio = self.slice(max(0, end - WINDOW), end)
        return np.pad(audio, (WINDOW - len(audio), 0))

    def reset(self):
        self.start, self.scored = None, None
        self.voiced = self.run = 0
        self.partial_at = 0

    def next_job(self):
        if self.language_job is not None:
            job, self.language_job = self.language_job, None
            return job
        if self.start is None:
            return None
        forced = self.total - self.start >= self.max_turn
        quiet = self.total - self.offset
        # Match the scene convention (PR #19): final speech end plus 0.3 s.
        end = self.start + self.max_turn if forced else self.offset + 4800
        if forced or quiet >= 4800:
            if self.scored is None or self.scored[0] != end:
                return ("score", end, None if forced else self.offset, self.window(end))
            _, turn, voice, window = self.scored
            if voice.value is not True:
                if forced or quiet >= self.timeout:
                    self.reset()
                return None
            reason = (
                "max_turn"
                if forced
                else "turn_end"
                if turn.probability >= self.threshold
                else "silence_timeout"
                if quiet >= self.timeout
                else None
            )
            if reason:
                audio = self.slice(self.start, end)
                future = self.slice(end, self.total)
                self.reset()
                if len(future):
                    self.total = end
                    self.audio = self.audio[: end - self.base]
                    pending, self.pending = self.pending, np.empty(0, dtype=np.float32)
                    self.ingest(future, self.captured_at)
                    self.pending = np.concatenate((self.pending, pending))
                self.finalizing = True
                return ("final", end, reason, audio, window, turn, voice)
        if (
            self.voiced >= RATE // 2
            and self.total > self.partial_at
            and perf_counter() - self.last_partial >= self.partial_seconds
        ):
            self.last_partial = perf_counter()
            self.partial_at = self.total
            return ("partial", self.total, self.slice(self.start, self.total))
        return None

    def compute(self, job):
        kind, end, *data = job
        if kind == "score":
            window = data[1]
            turn = self.model(window, [TURN], sampling_rate=RATE)[0]
            voice = self.model(window, [VOICE], sampling_rate=RATE)[0]
            if turn.probability is None:
                raise ValueError("Missing turn-end probability")
            return turn, voice
        if kind == "partial":
            return self.stt(data[0])
        if kind == "language":
            return self.model(data[0], [LANGUAGE], sampling_rate=RATE)[0]
        return self.stt(data[1])

    def complete(self, job, result):
        kind, end, *data = job
        if kind == "score":
            turn, voice = (replace(d, window_end_s=end / RATE) for d in result)
            self.emit(self.event("turn_end_score", [turn, voice]))
            if self.start is not None and (data[0] is None or self.offset == data[0]):
                self.scored = end, turn, voice, data[1]
            return None
        if kind == "partial":
            return self.response(result, "partial") if result else None
        if kind == "language":
            self.emit(
                {
                    **self.event(
                        "turn_decisions",
                        [data[1], data[2], replace(result, window_end_s=end / RATE)],
                    ),
                    **data[3],
                }
            )
            return None
        self.finalizing = False
        self.final_at = perf_counter()
        self.turn_count += 1
        endpoint_at = self.captured_at - (self.total + len(self.pending) - end) / RATE
        metadata = {
            "reason": data[0],
            "turn_count": self.turn_count,
            "stt_samples": len(data[1]),
            "endpoint_latency_ms": (perf_counter() - endpoint_at) * 1000,
        }
        self.language_job = ("language", end, data[2], data[3], data[4], metadata)
        return self.response(result, "final") if result else None

    @staticmethod
    def event(kind, decisions):
        return {
            "type": kind,
            "decisions": [
                {key: getattr(d, key) for key in ("task", "value", "probability", "window_end_s")}
                for d in decisions
            ],
        }

    @staticmethod
    def response(text, kind):
        return {
            "type": "transcriber-response",
            "transcription": text,
            "channel": "customer",
            "transcriptType": kind,
        }


async def session(websocket, model, stt, *, bridge=None, **options):
    await websocket.accept()
    message = await asyncio.wait_for(websocket.receive(), timeout=10)
    try:
        start = json.loads(message.get("text") or "null")
    except (ValueError, TypeError):
        start = None
    if (
        not isinstance(start, dict)
        or start.get("type") != "start"
        or start.get("encoding") != "linear16"
        or start.get("container") != "raw"
        or start.get("channels") != 2
        or type(start.get("sampleRate")) is not int
        or not 8000 <= start["sampleRate"] <= 48000
    ):
        await websocket.close(code=1003)
        return
    rate = start["sampleRate"]
    resampler = soxr.ResampleStream(rate, RATE, 1, dtype="float32") if rate != RATE else None
    bridge = bridge or Bridge(model, stt, **options)
    changed = asyncio.Event()
    ignored = 0

    async def receive():
        nonlocal ignored
        remainder = b""
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data is None:
                ignored += 1
                continue
            if len(data) > 1024 * 1024:
                raise BufferError("Frame too large")
            remainder += data
            size = len(remainder) // 4 * 4
            if not size:
                continue
            mono = np.frombuffer(remainder[:size], dtype="<i2")[::2].astype(np.float32) / 32768
            remainder = remainder[size:]
            if resampler is not None:
                mono = resampler.resample_chunk(mono)
            bridge.ingest(mono)
            changed.set()

    async def work():
        while True:
            changed.clear()
            job = bridge.next_job()
            if job is None:
                await changed.wait()
                continue
            result = await asyncio.to_thread(bridge.compute, job)
            frame = bridge.complete(job, result)
            if frame is not None:
                await websocket.send_json(frame)

    tasks = [asyncio.create_task(receive()), asyncio.create_task(work())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        bridge.emit(
            {
                "type": "session_closed",
                "ignored_messages": ignored,
                "samples": bridge.total,
                "turn_count": bridge.turn_count,
            }
        )


def create_app(model=None, stt=None, loader=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    def load():
        return (
            Marimba.from_pretrained(
                repo=os.getenv("MARIMBA_MODEL", "charcot-ai/marimba"), device="cpu", dtype="float32"
            ),
            whisper_stt(),
        )

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.model, app.state.stt = (
                (model, stt)
                if model is not None and stt is not None
                else await asyncio.to_thread(loader or load)
            )
        except Exception as error:
            app.state.model = app.state.stt = None
            logger.error(json.dumps({"type": "model_unavailable", "error": type(error).__name__}))
        yield

    app = FastAPI(lifespan=lifespan)
    calls = {}

    @app.get("/health")
    async def health():
        ready = getattr(app.state, "model", None) is not None
        return JSONResponse({"available": ready}, status_code=200 if ready else 503)

    @app.post("/endpointing/{key}")
    async def endpointing(key: str):
        bridge = calls.get(key)
        finished = (
            bridge is not None
            and bridge.start is None
            and not bridge.finalizing
            and 0 < perf_counter() - bridge.final_at < 2.5
        )
        return {"timeoutSeconds": 0.05 if finished else 5.0}

    @app.websocket("/transcriber")
    async def transcriber(websocket: WebSocket):
        key = websocket.query_params.get("session")
        if getattr(app.state, "model", None) is None or (key and key in calls):
            await websocket.accept()
            await websocket.close(code=1011)
            return
        bridge = None
        try:
            bridge = Bridge(
                app.state.model,
                app.state.stt,
                threshold=float(os.getenv("TURN_THRESHOLD", "0.5")),
                silence_timeout=float(os.getenv("SILENCE_TIMEOUT", "2.5")),
                max_turn_seconds=float(os.getenv("MAX_TURN_SECONDS", "30")),
            )
            if key:
                calls[key] = bridge
            await session(websocket, app.state.model, app.state.stt, bridge=bridge)
        except WebSocketDisconnect:
            pass
        except Exception as error:
            logger.error(json.dumps({"type": "session_error", "error": type(error).__name__}))
            with suppress(RuntimeError, OSError):
                await websocket.close(code=1013 if isinstance(error, BufferError) else 1011)
        finally:
            if key and calls.get(key) is bridge:
                calls.pop(key, None)

    return app


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
