import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from marimba import Question
from marimba.server import ServerConfig, create_app
from marimba.testing import FakeMarimba
from marimba.window_server import MAX_PAIR_WINDOWS, WINDOW_BYTES

ROOT = Path(__file__).parents[1]
OCTETS = {"Content-Type": "application/octet-stream"}
WINDOW_QUESTIONS = [
    {"task": "voice_activity", "seconds": 1, "threshold": 0.1},
    {"task": "turn_end"},
    {"task": "speaker_change", "kind": "score"},
    {"task": "speaker_count"},
]


def pcm(frequency: float = 220.0, windows: int = 1) -> bytes:
    t = np.arange(128000 * windows) / 16000
    return (0.2 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2").tobytes()


def post(
    client: TestClient, path: str, questions: Any, body: bytes, **headers: str
) -> tuple[int, dict[str, Any]]:
    sent = {**OCTETS, "Marimba-Questions": json.dumps(questions), **headers}
    response = client.post(path, content=body, headers=sent)
    return response.status_code, response.json()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeMarimba()))


def test_window_request_answers_each_question_in_order(client: TestClient) -> None:
    status, payload = post(client, "/v1/windows", WINDOW_QUESTIONS, pcm())
    assert status == 200
    assert payload["object"] == "window.decisions"
    assert payload["model"] == "charcot-ai/marimba"
    assert payload["latency_ms"] >= 0
    decisions = payload["decisions"]
    assert [d["task"] for d in decisions] == [q["task"] for q in WINDOW_QUESTIONS]
    assert [d["kind"] for d in decisions] == ["noul", "score", "score", "choice"]
    assert decisions[0]["threshold"] == 0.1
    assert (decisions[0]["window_start_s"], decisions[0]["window_end_s"]) == (7.0, 8.0)
    assert isinstance(decisions[0]["value"], bool)
    assert decisions[1]["value"] == decisions[1]["probability"]
    assert set(decisions[3]["probabilities"]) == {"0", "1", "2", "3+"}
    assert set(decisions[0]) == {
        "task",
        "kind",
        "value",
        "probability",
        "probabilities",
        "threshold",
        "status",
        "calibrated",
        "window_start_s",
        "window_end_s",
    }


def test_window_request_matches_the_python_api(client: TestClient) -> None:
    body = pcm()
    window = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    expected = FakeMarimba().decide_windows(
        [window], [Question.score("Has the speaker finished?")]
    )[0][0]
    _, payload = post(client, "/v1/windows", [{"task": "turn_end"}], body)
    assert payload["decisions"][0]["probability"] == expected.probability


@pytest.mark.parametrize("batched", [False, True])
def test_pair_request_answers_each_window_in_order(batched: bool) -> None:
    client = TestClient(create_app(FakeMarimba(), ServerConfig(batched_encoder=batched)))
    body = pcm(220.0) + bytes(WINDOW_BYTES) + pcm(220.0)
    question = [{"task": "same_speaker", "kind": "score"}]
    status, payload = post(client, "/v1/pairs", question, body)
    assert status == 200
    assert payload["object"] == "pair.decisions"
    values = [d["value"] for d in payload["decisions"]]
    assert len(values) == 3
    assert values[0] == values[2] != values[1]


@pytest.mark.parametrize(
    ("path", "questions", "body", "status", "code"),
    [
        ("/v1/windows", WINDOW_QUESTIONS, pcm()[:-2], 400, "invalid_audio_length"),
        ("/v1/windows", WINDOW_QUESTIONS, b"", 400, "invalid_audio_length"),
        ("/v1/windows", WINDOW_QUESTIONS, pcm(windows=2), 413, "request_too_large"),
        ("/v1/windows", [], pcm(), 400, "invalid_questions"),
        ("/v1/windows", {"task": "turn_end"}, pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "turn_end"}] * 9, pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "language"}], pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "turn_end", "text": "x"}], pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "turn_end", "kind": "choice"}], pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "turn_end", "threshold": 2}], pcm(), 400, "invalid_questions"),
        ("/v1/windows", [{"task": "turn_end", "seconds": 1}], pcm(), 400, "invalid_questions"),
        (
            "/v1/windows",
            [{"task": "voice_activity", "seconds": 3}],
            pcm(),
            400,
            "invalid_questions",
        ),
        (
            "/v1/windows",
            [{"task": "speaker_count", "threshold": 0.5}],
            pcm(),
            400,
            "invalid_questions",
        ),
        ("/v1/pairs", [{"task": "turn_end"}], pcm(), 400, "invalid_questions"),
        ("/v1/pairs", [{"task": "same_speaker"}] * 2, pcm(), 400, "invalid_questions"),
        ("/v1/pairs", [{"task": "same_speaker"}], pcm() + b"\0\0", 400, "invalid_audio_length"),
        (
            "/v1/pairs",
            [{"task": "same_speaker"}],
            bytes(WINDOW_BYTES * (MAX_PAIR_WINDOWS + 1)),
            413,
            "request_too_large",
        ),
    ],
)
def test_request_errors(
    client: TestClient, path: str, questions: Any, body: bytes, status: int, code: str
) -> None:
    got, payload = post(client, path, questions, body)
    assert (got, payload["error"]["code"]) == (status, code)
    assert payload["error"]["type"] == "invalid_request_error"


def test_header_and_media_type_errors(client: TestClient) -> None:
    missing = client.post("/v1/windows", content=pcm(), headers=OCTETS)
    assert (missing.status_code, missing.json()["error"]["code"]) == (400, "invalid_questions")
    malformed = client.post(
        "/v1/windows", content=pcm(), headers={**OCTETS, "Marimba-Questions": "[{"}
    )
    assert malformed.json()["error"]["code"] == "invalid_questions"
    status, payload = post(
        client, "/v1/windows", WINDOW_QUESTIONS, pcm(), **{"Content-Type": "application/json"}
    )
    assert (status, payload["error"]["code"]) == (415, "unsupported_media_type")


def test_bearer_token_and_capacity() -> None:
    secured = TestClient(create_app(FakeMarimba(), ServerConfig(api_key="synthetic-token")))
    status, payload = post(secured, "/v1/windows", WINDOW_QUESTIONS, pcm())
    assert (status, payload["error"]["code"]) == (401, "invalid_api_key")
    status, _ = post(
        secured, "/v1/pairs", [{"task": "speaker_change"}], pcm(), Authorization="Bearer nope"
    )
    assert status == 401
    status, _ = post(
        secured,
        "/v1/windows",
        WINDOW_QUESTIONS,
        pcm(),
        Authorization="Bearer synthetic-token",
    )
    assert status == 200
    assert secured.get("/health").status_code == 200


def test_inference_failure_returns_no_detail(caplog: pytest.LogCaptureFixture) -> None:
    class Broken(FakeMarimba):
        def decide_windows(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("synthetic secret detail")

    client = TestClient(create_app(Broken()))
    status, payload = post(client, "/v1/windows", WINDOW_QUESTIONS, pcm())
    assert (status, payload["error"]["code"]) == (500, "server_error")
    assert "synthetic secret detail" not in caplog.text + json.dumps(payload)
    # The slot is released after a failure.
    assert post(client, "/v1/windows", WINDOW_QUESTIONS, pcm())[0] == 500


def test_unavailable_task_is_rejected() -> None:
    class NoDiarization(FakeMarimba):
        @property
        def capabilities(self) -> Any:
            from dataclasses import replace

            caps = super().capabilities
            return {**caps, "same_speaker": replace(caps["same_speaker"], status="unavailable")}

    client = TestClient(create_app(NoDiarization()))
    status, payload = post(client, "/v1/pairs", [{"task": "same_speaker"}], pcm())
    assert (status, payload["error"]["code"]) == (400, "invalid_questions")


def test_contract_examples_match_the_server(client: TestClient) -> None:
    doc = (ROOT / "docs" / "window-server.md").read_text()
    for header in re.findall(r"-H 'Marimba-Questions: (\[.*?\])'", doc):
        questions = json.loads(header)
        path = "/v1/pairs" if len(questions) == 1 else "/v1/windows"
        assert post(client, path, questions, pcm())[0] == 200
    for constant in (str(WINDOW_BYTES), f"`1 <= N <= {MAX_PAIR_WINDOWS}`", "4096 bytes"):
        assert constant in doc.replace(",", "")
