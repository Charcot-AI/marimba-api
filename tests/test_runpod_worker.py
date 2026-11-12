"""Synthetic payload and worker lifecycle tests without Runpod or CUDA."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from marimba import Decision, Question

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "runpod", SimpleNamespace(serverless=Mock()))
    spec = importlib.util.spec_from_file_location("runpod_worker", ROOT / ".runpod/handler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def payload():
    return {
        "lane": "decoder",
        "sample_rate": 16000,
        "pcm16_b64": base64.b64encode(np.array([-32768, 0, 32767], dtype="<i2")).decode(),
        "mask_b64": base64.b64encode(bytes([0, 1, 1])).decode(),
        "window_start_s": 1.0,
        "window_end_s": 1.000125,
        "questions": [
            asdict(Question.choice("What language is being spoken?", ["English", "Portuguese"]))
        ],
    }


def test_import_does_not_load_model_or_start_server(worker):
    assert worker._load_model.cache_info().currsize == 0
    sys.modules["runpod"].serverless.start.assert_not_called()


def test_snapshot_preserves_pcm_scaling_mask_and_timestamps(worker, payload):
    snapshot = worker._snapshot(payload)
    np.testing.assert_array_equal(snapshot.audio, [-1.0, 0.0, 32767 / 32768])
    np.testing.assert_array_equal(snapshot.mask, [False, True, True])
    assert snapshot.audio.dtype == np.float32
    assert snapshot.window_start_s == payload["window_start_s"]
    assert snapshot.window_end_s == payload["window_end_s"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", 8000, "16000 Hz"),
        ("pcm16_b64", base64.b64encode(b"x").decode(), "odd byte count"),
        ("mask_b64", base64.b64encode(b"\x01").decode(), "lengths differ"),
        ("mask_b64", base64.b64encode(b"\x00\x00\x00").decode(), "valid audio"),
        ("mask_b64", base64.b64encode(b"\x00\x01\x02").decode(), "zero or one"),
        ("pcm16_b64", "!not-base64!", "base64"),
        ("mask_b64", "!not-base64!", "base64"),
        ("window_start_s", -1.0, "Invalid snapshot"),
        ("window_end_s", 0.0, "Invalid snapshot"),
        ("window_end_s", float("nan"), "Invalid snapshot"),
    ],
)
def test_invalid_snapshot_is_rejected(worker, payload, field, value, message):
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        worker._snapshot(payload)


@pytest.mark.parametrize("samples", [0, 30 * 16000 + 1])
def test_invalid_duration_is_rejected(worker, payload, samples):
    payload["pcm16_b64"] = base64.b64encode(bytes(samples * 2)).decode()
    payload["mask_b64"] = base64.b64encode(bytes([1]) * samples).decode()
    with pytest.raises(ValueError, match="valid audio"):
        worker._snapshot(payload)


def test_questions_restore_options_and_accept_term(worker, payload):
    values = payload["questions"] + [asdict(Question.noul('Did someone say "hello"?'))]
    questions = worker._questions(values)
    assert questions[0].options == ("English", "Portuguese")
    assert questions[1].term == "hello"


@pytest.mark.parametrize(
    "question",
    [Question.score("Is someone speaking?"), Question.noul("Is there any PII?")],
)
def test_other_tasks_are_rejected_before_model_loading(worker, payload, monkeypatch, question):
    load = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr(worker, "_load_model", load)
    payload["questions"].append(asdict(question))
    with pytest.raises(ValueError, match="Language and Term"):
        worker.handler({"input": payload})
    load.assert_not_called()


def test_empty_questions_are_rejected(worker):
    with pytest.raises(ValueError, match="At least one"):
        worker._questions([])


@pytest.mark.parametrize("lane", ["encoder", "invalid", None])
def test_non_decoder_lane_is_rejected(worker, payload, lane):
    payload["lane"] = lane
    with pytest.raises(ValueError, match="decoder-lane"):
        worker.handler({"input": payload})


def test_handler_returns_decisions_and_synchronized_timing(worker, payload, monkeypatch):
    question = worker._questions(payload["questions"])[0]
    decision = Decision(question, "language", "choice", "English", status="ok")
    decide = Mock(return_value=[decision])
    model = SimpleNamespace(lane=lambda question: "decoder", decide_snapshot=decide)
    monkeypatch.setattr(worker, "_load_model", lambda: model)
    synchronize = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
    )
    output = worker.handler({"input": payload})
    assert output["decisions"] == [decision.to_dict()]
    assert output["gpu_compute_ms"] >= 0
    assert synchronize.call_count == 2
    decide.assert_called_once()
    np.testing.assert_array_equal(decide.call_args.args[0].mask, [False, True, True])
    assert decide.call_args.args[1] == [question]


def test_startup_loads_once_with_validated_precision(worker, monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        set_float32_matmul_precision=Mock(),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    load = Mock(return_value=object())
    monkeypatch.setattr(worker.Marimba, "from_pretrained", load)
    worker.main()
    worker._load_model()
    load.assert_called_once_with(device="cuda", backend="torch")
    torch.set_float32_matmul_precision.assert_called_once_with("highest")
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    sys.modules["runpod"].serverless.start.assert_called_once_with({"handler": worker.handler})


def test_hub_smoke_test_is_valid_synthetic_silence(worker):
    tests = json.loads((ROOT / ".runpod/tests.json").read_text())
    data = tests["tests"][0]["input"]
    snapshot = worker._snapshot(data)
    assert len(snapshot.audio) == 16000
    assert not snapshot.audio.any()
    assert snapshot.mask.all()
    assert worker._questions(data["questions"])[0].task == "language"
