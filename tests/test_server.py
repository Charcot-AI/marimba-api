import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from marimba import Question
from marimba.server import (
    ServerConfig,
    count_audio_tokens,
    count_text_tokens,
    create_app,
    model_document,
    parse_questions,
)
from marimba.testing import FakeMarimba


def tone(seconds: float, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def wav_b64(seconds: float = 8.0, rate: int = 16000) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, tone(seconds, rate), rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode()


def body(text: str, data: str | None = None, fmt: str = "wav", **extra: Any) -> dict[str, Any]:
    return {
        "model": "charcot-ai/marimba",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data or wav_b64(), "format": fmt},
                    },
                ],
            }
        ],
        **extra,
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeMarimba()))


def test_shorthand_questions_route_to_tasks() -> None:
    questions = parse_questions('language? en, pt\nturn ended?\nterm: "marimba"\nspeech?')
    assert [q.task for q in questions] == ["language", "turn_end", "term", "voice_activity"]
    assert questions[0].options == ("en", "pt")
    assert [q.kind for q in questions] == ["choice", "score", "noul", "noul"]
    assert questions[2].term == "marimba"
    assert parse_questions("speaker count?")[0].options == ("0", "1", "2", "3+")
    assert len(parse_questions("language?")[0].options or ()) == 30


def test_json_and_template_questions() -> None:
    single = parse_questions('{"kind": "score", "text": "Is the turn over?", "threshold": 0.7}')
    assert single == [Question.score("Is the turn over?", threshold=0.7)]
    listed = parse_questions(
        '{"questions": [{"kind": "choice", "text": "Which language?", "options": ["fr", "de"]}]}'
    )
    assert listed[0].options == ("fr", "de")
    assert parse_questions("Has the speaker finished?")[0].kind == "score"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "what is the weather?",
        "term:",
        '{"text": "x"}',
        '[{"kind": "noul", "text": "x", "y": 1}]',
    ],
)
def test_invalid_questions(client: TestClient, text: str) -> None:
    response = client.post("/v1/chat/completions", json=body(text))
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


MARKER = "synthetic-marker-xyz"


@pytest.mark.parametrize(
    "text",
    [
        f'{{"kind": "noul", "text": "{MARKER}?", "task": "{MARKER}"}}',
        f'{{"kind": "choice", "text": "Is someone speaking?", "options": ["{MARKER}", "b"]}}',
        f'{{"kind": "noul", "text": "{MARKER}?", "task": "turn_end"}}',
        f'{{"kind": "noul", "text": "Did someone say it?", "task": "term", "x": "{MARKER}"}}',
        '{"kind": "noul", "text": "Did someone say it?", "task": "term"}',
        """{"kind": "noul", "text": "Did someone say '...'?", "task": "term"}""",
        f'{{"kind": "noul", "text": "Is someone speaking?", "options": ["{MARKER}"]}}',
        f'{{"kind": "{MARKER}", "text": "Is someone speaking?"}}',
        '{"kind": "noul", "text": "Is someone speaking?", "threshold": 2}',
        f'{{"kind": "choice", "text": "Which language?", "options": ["{MARKER}"]}}',
        f'{{"questions": "{MARKER}"}}',
        "[1, 2]",
        "[]",
        f"language? en, {MARKER}",
        "language? en, English",
        f"{MARKER}?",
        "\n".join(["speech?"] * 9),
    ],
)
def test_invalid_questions_are_rejected_without_logging_content(
    client: TestClient, caplog: pytest.LogCaptureFixture, text: str
) -> None:
    caplog.set_level(logging.DEBUG)
    response = client.post("/v1/chat/completions", json=body(text))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request_error"
    assert MARKER not in error["message"]
    assert not [r for r in caplog.records if r.exc_info or r.levelno >= logging.WARNING]
    assert MARKER not in caplog.text


@pytest.mark.parametrize(
    "request_body",
    [
        body("speech?", model=MARKER),
        body("speech?", stream=MARKER),
        body("speech?", data=base64.b64encode(MARKER.encode()).decode()),
        {"model": "charcot-ai/marimba", "messages": [{"role": "user", "content": [MARKER]}]},
    ],
)
def test_request_errors_do_not_echo_content(
    client: TestClient, caplog: pytest.LogCaptureFixture, request_body: dict[str, Any]
) -> None:
    caplog.set_level(logging.DEBUG)
    response = client.post("/v1/chat/completions", json=request_body)
    assert 400 <= response.status_code < 500
    assert MARKER not in response.text
    assert not [r for r in caplog.records if r.exc_info]
    assert MARKER not in caplog.text


class FailingMarimba(FakeMarimba):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(MARKER)


@pytest.mark.parametrize("stream", [False, True])
def test_inference_failure_logs_type_only(caplog: pytest.LogCaptureFixture, stream: bool) -> None:
    caplog.set_level(logging.DEBUG)
    failing = TestClient(create_app(FailingMarimba()), raise_server_exceptions=False)
    response = failing.post("/v1/chat/completions", json=body("speech?", stream=stream))
    assert MARKER not in response.text
    if stream:
        assert response.status_code == 200
        assert '"server_error"' in response.text
        assert "[DONE]" not in response.text
    else:
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"
    assert not [r for r in caplog.records if r.exc_info]
    assert MARKER not in caplog.text
    assert "Request inference failed: RuntimeError" in caplog.text
    ok = failing.get("/health")
    assert ok.status_code == 200


def test_token_accounting() -> None:
    assert count_audio_tokens(8.0) == 100
    assert count_audio_tokens(30.0) == 100
    assert count_audio_tokens(1.0) == 13
    assert count_audio_tokens(0.01) == 1
    assert count_text_tokens("term: marimba") == 3


def test_non_streaming_completion(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=body("language? en, pt\nturn ended?"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    decisions = json.loads(choice["message"]["content"])["decisions"]
    assert [d["task"] for d in decisions] == ["language", "turn_end"]
    assert decisions[0]["value"] in {"en", "pt"}
    assert decisions[1]["window_end_s"] == 8.0
    usage = payload["usage"]
    assert usage["prompt_tokens_details"]["audio_tokens"] == 100
    assert usage["prompt_tokens"] == 100 + count_text_tokens("language? en, pt\nturn ended?")
    assert usage["completion_tokens"] == count_text_tokens(choice["message"]["content"])
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_completion_matches_non_streaming(client: TestClient) -> None:
    request = body("term: marimba")
    plain = client.post("/v1/chat/completions", json=request).json()
    with client.stream("POST", "/v1/chat/completions", json={**request, "stream": True}) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in stream.iter_lines() if line]
    assert lines[0].startswith(":")
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[1:-1]]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == plain["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == plain["usage"]


def test_pcm16_and_resampled_audio(client: TestClient) -> None:
    pcm = (tone(2.0) * 32767).astype("<i2").tobytes()
    response = client.post(
        "/v1/chat/completions", json=body("speech?", base64.b64encode(pcm).decode(), "pcm16")
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens_details"]["audio_tokens"] == 25
    response = client.post("/v1/chat/completions", json=body("speech?", wav_b64(4.0, 44100)))
    assert response.json()["usage"]["prompt_tokens_details"]["audio_tokens"] == 50


@pytest.mark.parametrize(
    ("request_body", "status"),
    [
        (body("speech?", fmt="m4a"), 400),
        (body("speech?", data="not base64!"), 400),
        (body("speech?", data=wav_b64(31.0)), 413),
        (body("speech?", model="other/model"), 404),
        (body("speech?", n=2), 400),
        (
            {"model": "charcot-ai/marimba", "messages": [{"role": "user", "content": "speech?"}]},
            400,
        ),
    ],
)
def test_request_errors(client: TestClient, request_body: dict[str, Any], status: int) -> None:
    response = client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == status
    assert "message" in response.json()["error"]


def test_unavailable_task_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=body('{"kind": "noul", "text": "Is there PII?", "task": "pii"}'),
    )
    assert response.status_code == 400
    assert "pii" in response.json()["error"]["message"]


def test_api_key_and_capacity() -> None:
    secured = TestClient(create_app(FakeMarimba(), ServerConfig(api_key="synthetic-key")))
    assert secured.get("/v1/models").status_code == 401
    headers = {"Authorization": "Bearer synthetic-key"}
    assert secured.get("/v1/models", headers=headers).status_code == 200
    app = create_app(FakeMarimba(slow_decoder_seconds=1.0), ServerConfig(max_concurrency=1))
    with ThreadPoolExecutor(1) as pool:
        slow = pool.submit(TestClient(app).post, "/v1/chat/completions", json=body("term: x"))
        time.sleep(0.3)
        assert TestClient(app).post("/v1/chat/completions", json=body("speech?")).status_code == 429
        assert slow.result().status_code == 200
    assert TestClient(app).post("/v1/chat/completions", json=body("speech?")).status_code == 200


def test_models_document(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    (document,) = response.json()["data"]
    assert document["schema_version"] == "2.4"
    assert document["id"] == "charcot-ai/marimba"
    assert document["hugging_face_id"] == "charcot-ai/marimba"
    assert document["is_ready"] is False
    assert [m["type"] for m in document["input_modalities"]] == ["text", "audio"]
    (output,) = document["output_modalities"]
    assert output["type"] == "text" and output["streaming"] is True
    prices = [
        entry["cost_usd"]
        for modality in document["input_modalities"] + document["output_modalities"]
        for entry in modality["pricing"]
    ]
    assert prices and all(isinstance(price, str) for price in prices)
    text = document["input_modalities"][0]["supported_inputs"]["max_context_length"]
    assert text == {"value": 1024, "unit": "token"}
    custom = model_document(ServerConfig(datacenters=("US",), requests_per_minute=600))
    assert custom["datacenters"] == [{"country_code": "US"}]
    assert custom["capacity"] == {"value": 600, "unit": "request/minute"}


def test_health(client: TestClient) -> None:
    assert client.get("/ping").json() == {"status": "healthy"}
