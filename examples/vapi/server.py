"""Vapi custom-transcriber bridge; words come from an independent STT callable."""

import asyncio
import json
import os
from collections.abc import Callable
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
        self.stream = model.stream([], hop_seconds=0.25)
        self.pending = np.empty(0, dtype=np.float32)
        self.turn = np.empty(0, dtype=np.float32)
        self.speaking = False
        self.active = False
        self.last_partial = 0.0
        self.started = perf_counter()

    def feed(self, audio):
        self.pending = np.concatenate((self.pending, audio))
        frames = []
        while len(self.pending) >= 4000:
            chunk, self.pending = self.pending[:4000], self.pending[4000:]
            self.stream.ingest(chunk)
            self.turn = np.concatenate((self.turn, chunk))[-30 * 16000 :]
            tail = self.stream.snapshot(seconds=2.0)
            voice = self.model(tail.audio, [VOICE], sampling_rate=16000)[0]
            speaking = voice.value is True
            now = perf_counter()
            if speaking and not self.active:
                self.active = True
                self.started = now
            if speaking and now - self.last_partial >= self.partial_seconds:
                text = self.stt(self.turn.copy())
                if text:
                    frames.append(self.response(text, "partial"))
                self.last_partial = perf_counter()
            if self.speaking and not speaking and self.active:
                decision = self.stream.decide([TURN]).decisions[0]
                if decision.probability is None:
                    raise ValueError("Missing turn-end probability")
                if decision.probability >= self.threshold:
                    text = self.stt(self.turn.copy())
                    decisions = self.stream.decide([LANGUAGE]).decisions
                    frames.append(self.response(text, "final"))
                    self.emit(
                        {
                            "type": "turn_decisions",
                            "decisions": [d.to_dict() for d in [decision, *decisions]],
                            "turn_latency_ms": (perf_counter() - self.started) * 1000,
                        }
                    )
                    self.turn = np.empty(0, dtype=np.float32)
                    self.active = False
            self.speaking = speaking
            if not self.active:
                self.turn = self.turn[-32000:]
        return frames

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
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        if data is None:
            await websocket.close(code=1003)
            return
        remainder += data
        size = len(remainder) // 4 * 4
        if not size:
            continue
        mono = np.frombuffer(remainder[:size], dtype="<i2")[::2].astype(np.float32) / 32768
        remainder = remainder[size:]
        if resampler is not None:
            mono = resampler.resample_chunk(mono)
        for frame in await asyncio.to_thread(bridge.feed, mono):
            await websocket.send_json(frame)


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
