import struct
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from marimba_demo.cards import available_cards
from marimba_demo.protocol import decode_frame
from marimba_demo.server import create_app


@dataclass
class Runtime:
    backend: str = "test"
    device: str = "cpu"
    dtype: str = "float32"


class Question:
    @staticmethod
    def noul(text, *, task):
        return SimpleNamespace(text=text, task=task)


class ProtocolModel:
    """Contract double; never exposed by the demo launcher."""

    runtime = Runtime()

    def __init__(self):
        self.sessions = 0
        self.streams = []
        self.block = False
        self.created = threading.Event()
        self.capabilities = {
            "voice_activity": SimpleNamespace(status="validated", kinds=("noul",))
        }

    def stream(self, questions, **kwargs):
        self.sessions += 1
        assert kwargs == {
            "window_seconds": 8.0,
            "hop_seconds": 0.25,
            "sample_rate": 16000,
            "coalesce": True,
        }
        stream = ProtocolStream(self.block)
        self.streams.append(stream)
        self.created.set()
        return stream


class ProtocolStream:
    def __init__(self, block=False):
        self.count = 0
        self.decided = 0
        self.decide_calls = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.received_three = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def ingest(self, pcm):
        assert len(pcm) == 4000
        with self.lock:
            self.count += 1
            if self.count == 3:
                self.received_three.set()

    def decide(self):
        with self.lock:
            self.decide_calls += 1
            if self.count <= self.decided:
                return None
            count = self.count
            self.decided = count
        self.started.set()
        assert self.release.wait(5)
        return SimpleNamespace(window_end_s=count * 0.25, decisions=[])


@pytest.fixture
def client(tmp_path):
    (tmp_path / "index.html").write_text("<title>Test</title>")
    model = ProtocolModel()
    app = create_app(model, Question, static=tmp_path, local=tmp_path / "local")
    with TestClient(
        app, base_url="http://127.0.0.1:8765", headers={"host": "127.0.0.1:8765"}
    ) as client:
        yield client, model


def frame(sequence=0, timestamp=100.0):
    return (
        struct.pack("<Id", sequence, timestamp) + np.zeros(4000, dtype="<i2").tobytes()
    )


def test_pcm_is_little_endian():
    pcm = np.full(4000, -1234, dtype="<i2")
    result = decode_frame(struct.pack("<Id", 12, 800.5) + pcm.tobytes())
    assert result.sequence == 12
    assert result.captured_at_ms == 800.5
    assert result.pcm.tolist() == pcm.tolist()


@pytest.mark.parametrize(
    "data", [b"", b"x" * 8011, frame(timestamp=float("nan")), frame(timestamp=-1)]
)
def test_invalid_frame_rejected(data):
    with pytest.raises(ValueError):
        decode_frame(data)


def test_capability_gating(client):
    client, model = client
    assert [c.task for c in available_cards(model)] == ["voice_activity"]
    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json()["stt"] is False
    assert response.json()["cards"][0]["status"] == "validated"
    assert client.get("/local/stt_adapter.js").status_code == 404


def test_host_and_origin_protection(client):
    client, _ = client
    assert (
        client.get(
            "/api/local-config", headers={"host": "evil.example:8765"}
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/local-config", headers={"origin": "https://evil.example"}
        ).status_code
        == 403
    )
    with (
        pytest.raises(WebSocketDisconnect) as error,
        client.websocket_connect(
            "/api/stream", headers={"origin": "https://evil.example"}
        ),
    ):
        pass
    assert error.value.code == 1008


def test_decision_echoes_its_capture_timestamp(client):
    client, model = client
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_bytes(frame())
        assert socket.receive_json() == {
            "type": "update",
            "sequence": 0,
            "captured_at_ms": 100.0,
            "decision_interval_ms": 250.0,
            "coalesced_hops": 0,
            "decisions": [],
        }
        socket.send_bytes(frame(1, 350))
        assert socket.receive_json()["captured_at_ms"] == 350
        assert model.sessions == 1
        socket.send_json({"type": "stop"})
        assert socket.receive_json()["type"] == "stopped"
        assert model.streams[0].decide_calls == 2


def test_duplicate_sequence_is_rejected(client):
    client, _ = client
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_bytes(frame())
        socket.receive_json()
        socket.send_bytes(frame())
        assert socket.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1003


def test_missing_sequence_is_rejected(client):
    client, _ = client
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_bytes(frame())
        socket.receive_json()
        socket.send_bytes(frame(2))
        assert socket.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1003


def test_slow_decisions_preserve_all_ingested_audio_and_coalesce(client):
    client, model = client
    model.block = True
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_bytes(frame())
        assert model.created.wait(2)
        stream = model.streams[0]
        assert stream.started.wait(2)
        socket.send_bytes(frame(1, 350))
        socket.send_bytes(frame(2, 600))
        assert stream.received_three.wait(2)
        stream.release.set()
        assert socket.receive_json()["sequence"] == 0
        latest = socket.receive_json()
        assert latest["sequence"] == 2
        assert latest["captured_at_ms"] == 600
        assert latest["decision_interval_ms"] == 500
        assert latest["coalesced_hops"] == 1
        assert model.sessions == 1
        socket.send_json({"type": "stop"})
        assert socket.receive_json()["type"] == "stopped"


def test_invalid_json_control(client):
    client, _ = client
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_text("not-json")
        assert socket.receive_json()["type"] == "error"


def test_only_one_session_owns_model(client):
    client, _ = client
    headers = {"origin": "http://127.0.0.1:8765"}
    with (
        client.websocket_connect("/api/stream", headers=headers),
        client.websocket_connect("/api/stream", headers=headers) as second,
    ):
        assert second.receive_json()["type"] == "error"


def test_unavailable_and_wrong_kind_cards_are_omitted():
    model = ProtocolModel()
    model.capabilities.update(
        {
            "turn_end": SimpleNamespace(status="validated", kinds=("noul",)),
            "speaker_change": SimpleNamespace(status="unavailable", kinds=("noul",)),
            "same_speaker": SimpleNamespace(status="experimental", kinds=("noul",)),
        }
    )
    cards = available_cards(model)
    assert [(card.task, card.status) for card in cards] == [
        ("voice_activity", "validated"),
        ("same_speaker", "experimental"),
    ]


@pytest.mark.parametrize("origin", [None, "null"])
def test_websocket_requires_origin(client, origin):
    client, _ = client
    headers = {} if origin is None else {"origin": origin}
    with (
        pytest.raises(WebSocketDisconnect) as error,
        client.websocket_connect("/api/stream", headers=headers),
    ):
        pass
    assert error.value.code == 1008


def test_malformed_pcm_closes_socket(client):
    client, _ = client
    with client.websocket_connect(
        "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
    ) as socket:
        socket.send_bytes(b"invalid")
        assert socket.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1003


def test_transcript_config_is_not_embedded_in_public_bundle(client):
    client, _ = client
    response = client.get("/api/local-config")
    assert response.json() == {}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_real_contract_serialization_when_package_installed(tmp_path):
    package = pytest.importorskip("marimba")
    testing = pytest.importorskip("marimba.testing")
    (tmp_path / "index.html").write_text("<title>Test</title>")
    app = create_app(
        testing.FakeMarimba(), package.Question, static=tmp_path, local=tmp_path
    )
    with (
        TestClient(app, headers={"host": "127.0.0.1:8765"}) as client,
        client.websocket_connect(
            "/api/stream", headers={"origin": "http://127.0.0.1:8765"}
        ) as socket,
    ):
        socket.send_bytes(frame())
        update = socket.receive_json()
        assert update["type"] == "update", update
        assert update["decisions"]
        assert all(
            "latency_ms" in d and "window_end_s" in d for d in update["decisions"]
        )
