"""Vapi custom-transcriber bridge; words come from an independent STT callable."""

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

import numpy as np
import soxr

from marimba import Marimba, Question

VOICE = Question.noul("Is someone speaking?", task="voice_activity")
TURN = Question.score("Has the speaker finished their turn?", task="turn_end")
LANGUAGE = Question.choice("What language is being spoken?", ("en", "pt", "es"), task="language")


def whisper_stt() -> Callable[[np.ndarray], str]:
    from transformers import pipeline

    transcribe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    return lambda audio: transcribe({"raw": audio, "sampling_rate": 16000})["text"].strip()


class Bridge:
    def __init__(self, model, stt, *, threshold=0.5, partial_seconds=1.0, emit=None):
        if not 0 <= threshold <= 1 or partial_seconds < 1:
            raise ValueError("Invalid threshold or partial interval")
        self.model, self.stt = model, stt
        self.threshold, self.partial_seconds = threshold, partial_seconds
        self.emit = emit or (lambda event: print(json.dumps(event), flush=True))
        self.ring = np.empty(0, dtype=np.float32)
        self.total = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.turn = np.empty(0, dtype=np.float32)
        self.speaking = False
        self.active = False
        self.last_partial = 0.0

    def feed(self, audio, captured_at=None):
        captured_at = perf_counter() if captured_at is None else captured_at
        self.pending = np.concatenate((self.pending, audio))
        count = len(self.pending) // 4000 * 4000
        if not count:
            return []
        chunk, self.pending = self.pending[:count], self.pending[count:]
        self.total += count
        self.ring = np.concatenate((self.ring, chunk))[-160000:]
        self.turn = np.concatenate((self.turn, chunk))[-30 * 16000 :]
        voice = self.model(self.ring[-32000:], [VOICE], sampling_rate=16000)[0]
        speaking = voice.value is True
        frames = []
        if speaking:
            self.active = True
        if speaking and perf_counter() - self.last_partial >= self.partial_seconds:
            text = self.stt(self.turn.copy())
            if text:
                frames.append(self.response(text, "partial"))
            self.last_partial = perf_counter()
        if self.speaking and not speaking and self.active:
            offset = max(0, self.total - 28000)
            ring_start = self.total - len(self.ring)
            window = self.ring[max(0, offset - 128000 - ring_start) : offset - ring_start]
            if len(window):
                decision = self.model(window, [TURN], sampling_rate=16000)[0]
                decision = replace(decision, window_end_s=offset / 16000)
                if decision.probability is None:
                    raise ValueError("Missing turn-end probability")
                self.emit(self.event("turn_end_score", [decision]))
                if decision.probability >= self.threshold:
                    text = self.stt(self.turn.copy())
                    language = self.model(window, [LANGUAGE], sampling_rate=16000)[0]
                    language = replace(language, window_end_s=offset / 16000)
                    frames.append(self.response(text, "final"))
                    offset_at = captured_at - (len(self.pending) + self.total - offset) / 16000
                    self.emit(
                        {
                            **self.event("turn_decisions", [decision, language]),
                            "endpoint_latency_ms": (perf_counter() - offset_at) * 1000,
                        }
                    )
                    self.turn = np.empty(0, dtype=np.float32)
                    self.active = False
        self.speaking = speaking
        if not self.active:
            self.turn = self.turn[-32000:]
        return frames

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


async def session(websocket, model, stt, **options):
    await websocket.accept()
    start = json.loads(await websocket.receive_text())
    rate = start.get("sampleRate")
    if (
        start.get("type") != "start"
        or start.get("encoding") != "linear16"
        or start.get("container") != "raw"
        or start.get("channels") != 2
        or type(rate) is not int
        or not 8000 <= rate <= 48000
    ):
        await websocket.close(code=1003)
        return
    resampler = soxr.ResampleStream(rate, 16000, 1, dtype="float32") if rate != 16000 else None
    bridge = Bridge(model, stt, **options)
    remainder = b""
    queue = asyncio.Queue()

    async def receive():
        nonlocal remainder
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                await queue.put(None)
                return
            data = message.get("bytes")
            if data is None:
                await websocket.close(code=1003)
                await queue.put(None)
                return
            remainder += data
            size = len(remainder) // 4 * 4
            if not size:
                continue
            mono = np.frombuffer(remainder[:size], dtype="<i2")[::2].astype(np.float32) / 32768
            remainder = remainder[size:]
            if resampler is not None:
                mono = resampler.resample_chunk(mono)
            await queue.put((mono, perf_counter()))

    receiver = asyncio.create_task(receive())
    try:
        finished = False
        while not finished:
            item = await queue.get()
            if item is None:
                break
            chunks = [item[0]]
            captured_at = item[1]
            while not queue.empty():
                item = queue.get_nowait()
                if item is None:
                    finished = True
                    break
                chunks.append(item[0])
                captured_at = item[1]
            for frame in await asyncio.to_thread(bridge.feed, np.concatenate(chunks), captured_at):
                await websocket.send_json(frame)
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)


def create_app(model=None, stt=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app = FastAPI()
    lock = asyncio.Lock()

    @app.websocket("/transcriber")
    async def transcriber(websocket: WebSocket):
        async with lock:
            try:
                await session(
                    websocket,
                    model or app.state.model,
                    stt or app.state.stt,
                    threshold=float(os.getenv("TURN_THRESHOLD", "0.5")),
                )
            except WebSocketDisconnect:
                pass
            except (ValueError, RuntimeError):
                await websocket.close(code=1011)

    return app


def main():
    import uvicorn

    app = create_app()
    app.state.model = Marimba.from_pretrained(device="cpu", dtype="float32")
    app.state.stt = whisper_stt()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
