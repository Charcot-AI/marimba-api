import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from marimba.testing import FakeMarimba

spec = importlib.util.spec_from_file_location(
    "vapi_server", Path(__file__).parents[1] / "examples/vapi/server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class Socket:
    def __init__(self, rate, audio, **overrides):
        self.start = dict(
            type="start",
            encoding="linear16",
            container="raw",
            sampleRate=rate,
            channels=2,
            **overrides,
        )
        raw = audio.astype("<i2").tobytes()
        self.messages = [
            {"type": "websocket.receive", "bytes": raw[i : i + 997]}
            for i in range(0, len(raw), 997)
        ]
        self.messages.append({"type": "websocket.disconnect"})
        self.frames = []
        self.closed = None

    async def accept(self):
        pass

    async def receive_text(self):
        return json.dumps(self.start)

    async def receive(self):
        return self.messages.pop(0)

    async def send_json(self, frame):
        self.frames.append(frame)

    async def close(self, code):
        self.closed = code


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
def test_session(rate):
    tone = 30000 * np.sign(np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate))
    customer = np.concatenate((tone, np.zeros(rate * 3)))
    stereo = np.column_stack((customer, np.full(len(customer), -32000)))
    socket = Socket(rate, stereo)
    seen, events = [], []

    def stt(audio):
        seen.append(audio)
        return "Synthetic hello"

    asyncio.run(server.session(socket, FakeMarimba(), stt, threshold=0.01, emit=events.append))
    assert [frame["transcriptType"] for frame in socket.frames] == ["partial", "final"]
    for frame in socket.frames:
        assert frame == dict(
            type="transcriber-response",
            transcription="Synthetic hello",
            channel="customer",
            transcriptType=frame["transcriptType"],
        )
    assert len(events) == 1
    assert events[0]["decisions"][0]["task"] == "turn_end"
    assert np.max(np.abs(seen[-1][-16000:])) < 0.01
    assert max(seen[-1]) > 0.8


def test_empty_and_assistant_only():
    for count in (0, 64000):
        socket = Socket(16000, np.column_stack((np.zeros(count), np.full(count, 32000))))
        asyncio.run(server.session(socket, FakeMarimba(), lambda _: pytest.fail("STT called")))
        assert socket.frames == []


def test_invalid_start_and_text_frame():
    socket = Socket(16000, np.empty((0, 2)))
    socket.start["encoding"] = "mulaw"
    asyncio.run(server.session(socket, FakeMarimba(), lambda _: ""))
    assert socket.closed == 1003
    socket = Socket(16000, np.empty((0, 2)))
    socket.messages.insert(0, {"type": "websocket.receive", "text": "unexpected"})
    asyncio.run(server.session(socket, FakeMarimba(), lambda _: ""))
    assert socket.closed == 1003


def test_stt_failure():
    def fail(_):
        raise RuntimeError("Synthetic STT failure")

    bridge = server.Bridge(FakeMarimba(), fail)
    with pytest.raises(RuntimeError, match="Synthetic STT failure"):
        bridge.feed(np.ones(4000, dtype=np.float32))
