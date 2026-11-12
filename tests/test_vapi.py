import asyncio
import importlib.util
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marimba.testing import FakeMarimba

spec = importlib.util.spec_from_file_location(
    "vapi_server", Path(__file__).parents[1] / "examples/vapi/server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
START = dict(type="start", encoding="linear16", container="raw", sampleRate=16000, channels=2)


class Model(FakeMarimba):
    def __init__(self, score=0.9):
        super().__init__()
        self.score = score
        self.calls = []

    def __call__(self, audio, questions, sampling_rate=None):
        self.calls.append((questions[0].task, audio.copy()))
        result = super().__call__(audio, questions, sampling_rate)
        return [
            replace(d, value=self.score, probability=self.score) if d.task == "turn_end" else d
            for d in result
        ]


def speech(seconds=1, value=0.2):
    return np.full(round(seconds * 16000), value, dtype=np.float32)


def silence(seconds):
    return np.zeros(round(seconds * 16000), dtype=np.float32)


def drain(bridge):
    frames = []
    while (job := bridge.next_job()) is not None:
        frame = bridge.complete(job, bridge.compute(job))
        if frame:
            frames.append(frame)
    return frames


class Socket:
    def __init__(self, start=None):
        self.messages = asyncio.Queue()
        self.messages.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(START if start is None else start)}
        )
        self.frames, self.closed = [], None

    async def accept(self):
        pass

    async def receive(self):
        item = await self.messages.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code):
        self.closed = code

    async def send_json(self, frame):
        self.frames.append(frame)

    def send(self, audio, rate=16000):
        raw = np.column_stack((audio * 32767, np.full(len(audio), -30000))).astype("<i2").tobytes()
        for i in range(0, len(raw), 997):
            self.messages.put_nowait({"type": "websocket.receive", "bytes": raw[i : i + 997]})


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 3)


def test_exact_window_endpoint_and_final_audio():
    model, seen, events = Model(), [], []
    bridge = server.Bridge(
        model, lambda audio: seen.append(audio.copy()) or "Synthetic words", emit=events.append
    )
    audio = np.concatenate((silence(8), speech(1), silence(0.3)))
    bridge.ingest(audio)
    frames = drain(bridge)
    assert frames == [server.Bridge.response("Synthetic words", "final")]
    expected = audio[-128000:]
    assert [task for task, _ in model.calls] == ["turn_end", "language"]
    for _, window in model.calls:
        np.testing.assert_array_equal(window, expected)
    np.testing.assert_array_equal(seen[0], audio[round(7.8 * 16000) :])
    assert events[0]["decisions"][0]["window_end_s"] == 9.3
    assert events[1]["reason"] == "turn_end"


def test_timeout_and_resume_continues_turn():
    events = []
    bridge = server.Bridge(Model(score=0.1), lambda _: "Synthetic words", emit=events.append)
    bridge.ingest(speech())
    assert drain(bridge)[0]["transcriptType"] == "partial"
    bridge.ingest(silence(0.3))
    assert drain(bridge) == []
    start = bridge.start
    bridge.ingest(speech())
    assert bridge.start == start
    bridge.ingest(silence(0.3))
    assert drain(bridge) == []
    bridge.ingest(silence(2.2))
    assert drain(bridge)[0]["transcriptType"] == "final"
    assert events[-1]["reason"] == "silence_timeout"


def test_no_loss_while_final_in_flight():
    seen = []
    bridge = server.Bridge(Model(), lambda audio: seen.append(audio.copy()) or "Synthetic words")
    first = np.concatenate((speech(), silence(0.3)))
    bridge.ingest(first)
    score = bridge.next_job()
    bridge.complete(score, bridge.compute(score))
    final = bridge.next_job()
    second = np.concatenate((speech(1, 0.4), silence(0.3)))
    bridge.ingest(second)
    bridge.complete(final, bridge.compute(final))
    drain(bridge)
    np.testing.assert_array_equal(seen[0], first)
    np.testing.assert_array_equal(seen[1][-len(second) :], second)
    assert bridge.turn_count == 2


def test_stale_score_cannot_cut_resumed_speech():
    bridge = server.Bridge(Model(), lambda _: "Synthetic words")
    bridge.ingest(np.concatenate((speech(), silence(0.3))))
    job = bridge.next_job()
    bridge.ingest(speech())
    bridge.complete(job, bridge.compute(job))
    assert bridge.scored is None
    assert bridge.start == 0
    assert all(x["transcriptType"] == "partial" for x in drain(bridge))


def test_delayed_endpoint_retains_exact_history():
    bridge = server.Bridge(Model(), lambda _: "Synthetic words")
    background = np.full(10 * 16000, 0.001, dtype=np.float32)
    bridge.ingest(np.concatenate((background, speech())))
    partial = bridge.next_job()
    assert partial[0] == "partial"
    bridge.ingest(silence(0.8))
    bridge.complete(partial, bridge.compute(partial))
    expected = np.concatenate((background, speech(), silence(0.3)))[-128000:]
    score = bridge.next_job()
    np.testing.assert_array_equal(score[3], expected)
    bridge.ingest(silence(1))
    np.testing.assert_array_equal(bridge.window(score[1]), expected)
    bridge.complete(score, bridge.compute(score))
    final = bridge.next_job()
    bridge.ingest(silence(40))
    np.testing.assert_array_equal(final[4], expected)
    assert len(bridge.audio) == server.WINDOW
    assert len(bridge.audio) <= bridge.max_turn + server.WINDOW
    bridge.complete(final, bridge.compute(final))
    assert bridge.pinned is None


def test_final_metadata_survives_immediate_disconnect(caplog):
    caplog.set_level(logging.INFO, logger="vapi_bridge")

    async def run():
        class DisconnectAfterFinal(Socket):
            async def send_json(self, frame):
                if frame["transcriptType"] == "final":
                    events = [json.loads(record.message) for record in caplog.records]
                    decision = events[-1]
                    assert decision["type"] == "turn_decisions"
                    assert decision["reason"] == "turn_end"
                    assert [d["task"] for d in decision["decisions"]] == [
                        "turn_end",
                        "language",
                    ]
                    self.messages.put_nowait({"type": "websocket.disconnect"})
                await super().send_json(frame)

        socket = DisconnectAfterFinal()
        socket.send(np.concatenate((speech(), silence(0.3))))
        await asyncio.wait_for(server.session(socket, Model(), lambda _: "Synthetic words"), 3)
        assert socket.frames[-1]["transcriptType"] == "final"
        assert sum('"type": "turn_decisions"' in record.message for record in caplog.records) == 1

    asyncio.run(run())


def test_language_failure_cannot_emit_incomplete_final():
    class FailedLanguage(Model):
        def __call__(self, audio, questions, sampling_rate=None):
            if questions[0].task == "language":
                raise RuntimeError("Synthetic language failure")
            return super().__call__(audio, questions, sampling_rate)

    events = []
    bridge = server.Bridge(FailedLanguage(), lambda _: "Synthetic words", emit=events.append)
    bridge.ingest(np.concatenate((speech(), silence(0.3))))
    with pytest.raises(RuntimeError, match="Synthetic language failure"):
        drain(bridge)
    assert bridge.turn_count == 0
    assert all(event["type"] != "turn_decisions" for event in events)


@pytest.mark.parametrize("score", [0.1, 0.9])
def test_empty_stt_and_silence_memory(score):
    bridge = server.Bridge(Model(score=score), lambda _: "")
    bridge.ingest(np.concatenate((speech(), silence(0.3))))
    assert drain(bridge) == []
    for _ in range(600):
        bridge.ingest(silence(1))
        assert drain(bridge) == []
    assert len(bridge.audio) == 128000
    assert bridge.start is None
    assert bridge.turn_count == 0
    assert bridge.final_at == 0
    assert drain(bridge) == []


def test_short_energy_spike_does_not_start_turn():
    bridge = server.Bridge(Model(), lambda _: pytest.fail("STT called"))
    bridge.ingest(np.concatenate((speech(0.04), silence(1))))
    assert bridge.start is None
    assert drain(bridge) == []


def test_resumption_preserves_audio_after_low_turn_score():
    model, seen = Model(score=0.1), []
    bridge = server.Bridge(model, lambda audio: seen.append(audio.copy()) or "Synthetic words")
    first = np.concatenate((speech(), silence(0.5)))
    bridge.ingest(first)
    assert drain(bridge)[0]["transcriptType"] == "partial"
    assert bridge.start == 0
    seen.clear()
    model.score = 0.9
    second = np.concatenate((speech(1, 0.4), silence(0.3)))
    bridge.ingest(second)
    assert drain(bridge)[0]["transcriptType"] == "final"
    np.testing.assert_array_equal(seen[0], np.concatenate((first, second)))


def test_max_turn_bounds_and_retains_future():
    events, seen = [], []
    bridge = server.Bridge(
        Model(),
        lambda audio: seen.append(audio.copy()) or "Synthetic words",
        max_turn_seconds=1,
        emit=events.append,
    )
    bridge.ingest(speech(1.5))
    assert drain(bridge)[0]["transcriptType"] == "final"
    assert events[-1]["reason"] == "max_turn"
    assert len(seen[0]) == 16000
    assert bridge.start is not None
    with pytest.raises(BufferError):
        bridge.ingest(speech(10))


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
def test_session_resampling_fragments_and_schema(rate):
    async def run():
        socket = Socket({**START, "sampleRate": rate})
        task = asyncio.create_task(server.session(socket, Model(), lambda _: "Synthetic words"))
        socket.send(np.full(rate, 0.2))
        await until(lambda: socket.frames)
        socket.send(np.zeros(rate))
        await until(lambda: any(x["transcriptType"] == "final" for x in socket.frames))
        socket.messages.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(task, 1)
        for frame in socket.frames:
            assert frame == server.Bridge.response("Synthetic words", frame["transcriptType"])

    asyncio.run(run())


def test_receive_does_not_wait_for_slow_stt_and_disconnect():
    async def run():
        started, release = threading.Event(), threading.Event()

        def stt(_):
            started.set()
            release.wait(2)
            return "Synthetic words"

        bridge = server.Bridge(Model(), stt)
        socket = Socket()
        task = asyncio.create_task(server.session(socket, bridge.model, stt, bridge=bridge))
        socket.send(speech(0.6))
        await until(started.is_set)
        socket.send(speech(0.6))
        await until(lambda: bridge.total >= 19200)
        assert not socket.frames
        socket.messages.put_nowait({"type": "websocket.disconnect"})
        try:
            await asyncio.wait_for(task, 0.2)
        finally:
            release.set()

    asyncio.run(run())


@pytest.mark.parametrize(
    "start", [[], None, "bad", {**START, "encoding": "mulaw"}, {**START, "channels": 1}]
)
def test_bad_start(start):
    async def run():
        socket = Socket()
        socket.messages = asyncio.Queue()
        socket.messages.put_nowait({"type": "websocket.receive", "text": json.dumps(start)})
        await asyncio.wait_for(server.session(socket, Model(), lambda _: ""), 1)
        assert socket.closed == 1003

    asyncio.run(run())


def test_binary_before_start_and_malformed_json():
    async def run():
        for message in ({"bytes": b"\0"}, {"text": "{"}):
            socket = Socket()
            socket.messages = asyncio.Queue()
            socket.messages.put_nowait({"type": "websocket.receive", **message})
            await server.session(socket, Model(), lambda _: "")
            assert socket.closed == 1003

    asyncio.run(run())


def test_unknown_text_and_receiver_failure():
    async def run():
        events = []
        socket = Socket()
        socket.messages.put_nowait({"type": "websocket.receive", "text": "{"})
        socket.messages.put_nowait({"type": "websocket.receive", "text": '{"type":"unknown"}'})
        socket.messages.put_nowait(ValueError("Synthetic receive failure"))
        with pytest.raises(ValueError, match="Synthetic receive failure"):
            await asyncio.wait_for(
                server.session(socket, Model(), lambda _: "", emit=events.append), 1
            )
        assert events[-1]["ignored_messages"] == 2

    asyncio.run(run())


def test_log_hygiene_and_worker_failure(caplog):
    caplog.set_level(logging.INFO, logger="vapi_bridge")
    secret_text = "Synthetic transcript sentinel"
    bridge = server.Bridge(Model(), lambda _: secret_text)
    bridge.ingest(np.concatenate((speech(), silence(0.3))))
    assert drain(bridge)[0]["transcription"] == secret_text
    assert secret_text not in caplog.text
    assert "transcription" not in caplog.text
    assert "audio" not in caplog.text

    def fail(_):
        raise RuntimeError(secret_text)

    async def run():
        socket = Socket()
        socket.send(speech())
        with pytest.raises(RuntimeError, match=secret_text):
            await asyncio.wait_for(server.session(socket, Model(), fail), 1)

    asyncio.run(run())


def test_app_failure_health_and_endpointing(caplog):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    caplog.set_level(logging.INFO, logger="vapi_bridge")

    def fail():
        raise RuntimeError("Synthetic private error detail")

    with TestClient(server.create_app(loader=fail)) as client:
        assert client.get("/health").status_code == 503
        with client.websocket_connect("/transcriber") as ws:
            assert ws.receive()["code"] == 1011
    assert "Synthetic private error detail" not in caplog.text
    assert caplog.text.count('"type": "model_unavailable"') == 1
    with TestClient(server.create_app(Model(), lambda _: "Synthetic words")) as client:
        assert client.get("/health").json() == {"available": True}
        assert client.post("/endpointing/missing", json={}).json() == {"timeoutSeconds": 5.0}
        with client.websocket_connect("/transcriber?session=test") as ws:
            ws.send_json(START)
            ws.send_text("unknown")
            audio = np.concatenate((speech(), silence(1)))
            ws.send_bytes(
                np.column_stack((audio * 32767, np.zeros(len(audio)))).astype("<i2").tobytes()
            )
            while ws.receive_json()["transcriptType"] != "final":
                pass
            assert client.post("/endpointing/test", json={}).json() == {"timeoutSeconds": 0.05}
        assert client.post("/endpointing/test", json={}).json() == {"timeoutSeconds": 5.0}

    def fail_stt(_):
        raise RuntimeError("Synthetic transcript sentinel")

    with TestClient(server.create_app(Model(), fail_stt)) as client:
        with client.websocket_connect("/transcriber") as ws:
            ws.send_json(START)
            audio = speech()
            ws.send_bytes(
                np.column_stack((audio * 32767, np.zeros(len(audio)))).astype("<i2").tobytes()
            )
            assert ws.receive()["code"] == 1011
    assert "Synthetic transcript sentinel" not in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": float("nan")},
        {"silence_timeout": 0},
        {"max_turn_seconds": 0},
        {"partial_seconds": 0},
    ],
)
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        server.Bridge(Model(), lambda _: "", **options)
