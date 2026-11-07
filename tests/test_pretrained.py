import hashlib
import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from marimba import Marimba, Question, RuntimeInfo, UnsupportedQuestion
from marimba.audio import right_aligned_window
from marimba.lexical import exact_fraction, normalize
from marimba.pretrained import PretrainedMarimba
from marimba.readout import pool_layer, predict_readout
from marimba.weights import DEFAULT_MODEL_REVISION, ENCODER_REPO, ENCODER_REVISION, load_manifest


class StubEngine:
    def __init__(self):
        self.calls = []

    def encode(self, window):
        self.calls.append("encode")
        return object()

    def turn_end(self, mask):
        self.calls.append("turn_end")
        return 0.6, 0.7

    def hypotheses(self, encoded):
        self.calls.append("hypotheses")
        return ["Hello world"] * 4 + ["World"]

    def language_scores(self, encoded, options):
        self.calls.append("language")
        return {option: -float(index + 1) for index, option in enumerate(options)}

    def head_score(self, mask, text):
        self.calls.append("head")
        return 2.0

    term_score = staticmethod(exact_fraction)


@pytest.fixture
def model():
    manifest = {
        "resolved_revision": DEFAULT_MODEL_REVISION,
        "lanes": {
            "turn_end": {"status": "validated"},
            "language": {"status": "validated"},
            "term": {
                "status": "experimental",
                "calibration": "none",
                "evaluation": "Uncalibrated fraction",
            },
        },
    }
    return PretrainedMarimba(RuntimeInfo("fake", "cpu", "float32"), StubEngine(), manifest)


def test_shared_lanes(model):
    questions = [
        Question.score("Has the speaker finished?"),
        Question.choice("What language?", ["en", "pt"]),
        Question.noul('Did someone say "hello"?'),
        Question.noul('Did someone say "absent"?'),
    ]
    decisions = model(np.zeros(16000, dtype=np.float32), questions, 16000)
    assert model._engine.calls == ["encode", "turn_end", "language", "hypotheses"]
    assert decisions[0].probability == 0.7
    assert decisions[0].score == 0.6
    assert decisions[1].value == "en"
    assert sum(decisions[1].probabilities.values()) == pytest.approx(1)
    assert [decision.probability for decision in decisions[2:]] == [0.8, 0.0]
    assert all(json.loads(json.dumps(d.to_dict())) == d.to_dict() for d in decisions)
    assert "Uncalibrated" in model.capabilities["term"].evaluation


def test_unavailable_and_empty(model):
    question = Question.noul("Is someone speaking?")
    decision = model(np.zeros(1, dtype=np.float32), [question], 16000)[0]
    assert decision.status == "unavailable"
    assert decision.value is None and decision.probability is None
    assert model._engine.calls == []
    assert model(np.zeros(1, dtype=np.float32), [], 16000) == []
    with pytest.raises(ValueError, match="nonempty"):
        model(np.array([], dtype=np.float32), [question], 16000)


def test_shared_speech_head(model):
    model._capabilities["voice_activity"] = replace(
        model.capabilities["voice_activity"], status="validated"
    )
    model._capabilities["same_speaker"] = replace(
        model.capabilities["same_speaker"], status="experimental"
    )
    questions = [
        Question.score("Has the speaker finished?"),
        Question.noul("Is someone speaking?"),
        Question.noul("Is this the same speaker?"),
        Question.score("Is this the same speaker?"),
        Question.choice("How many speakers?", ["0", "1", "2"]),
    ]
    decisions = model(np.zeros(16000, dtype=np.float32), questions, 16000)
    assert model._engine.calls == ["encode", "turn_end", "head", "head", "head"]
    assert [d.status for d in decisions] == [
        "validated",
        "validated",
        "experimental",
        "experimental",
        "unavailable",
    ]
    assert decisions[1].score == 2.0
    assert decisions[1].probability == pytest.approx(1 / (1 + np.exp(-2)))
    assert decisions[1].value is True
    assert decisions[3].value == decisions[3].probability
    assert decisions[4].value is None


def test_validation_before_encoder(model):
    for question, error in [
        (Question.choice("What language?", ["en", "xx"]), ValueError),
        (Question.noul("Custom", task="unknown"), UnsupportedQuestion),
        (Question.noul("What language?"), UnsupportedQuestion),
        (Question.noul("Custom", task="term"), UnsupportedQuestion),
    ]:
        with pytest.raises(error):
            model(np.zeros(1, dtype=np.float32), [question], 16000)
    assert model._engine.calls == []


def test_encoder_failure_propagates(model, monkeypatch):
    def fail(window):
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(model._engine, "encode", fail)
    with pytest.raises(RuntimeError, match="synthetic backend"):
        model(np.zeros(1, dtype=np.float32), [Question.score("Has the speaker finished?")], 16000)


def test_lexical_normalization():
    assert Question.noul('Did someone say "don\'t"?').term == "don't"
    assert normalize("Café, WORLD_你好!") == "cafe world 你好"
    assert exact_fraction("café", ["a cafe here", "cafeteria", "CAFÉ", "", "tea"]) == 0.4
    assert exact_fraction("hello world", ["Hello, world!", "hello worlds"]) == 0.5
    for term, hypotheses in [("", ["hello"]), ("hello", [])]:
        with pytest.raises(ValueError):
            exact_fraction(term, hypotheses)


def test_pool_and_readout():
    _, mask = right_aligned_window(np.ones(16000, dtype=np.float32))
    frames = np.arange(32, dtype=np.float32).reshape(16, 2)
    pooled = pool_layer(frames, mask)
    np.testing.assert_equal(pooled, frames[-2:].mean(axis=0))
    weights = {
        "mean": np.array([0.0, 0.0]),
        "scale": np.array([2.0, 2.0]),
        "coef": np.array([0.1, 0.2]),
        "intercept": np.array([-1.0]),
        "platt_coef": np.array([0.5]),
        "platt_intercept": np.array([-0.1]),
    }
    raw, calibrated = predict_readout(weights, pooled)
    logit = (pooled / weights["scale"]) @ weights["coef"] - 1
    assert raw == pytest.approx(1 / (1 + np.exp(-logit)))
    assert calibrated == pytest.approx(1 / (1 + np.exp(-(logit * 0.5 - 0.1))))
    with pytest.raises(ValueError, match="valid"):
        pool_layer(frames, np.zeros(128000, dtype=bool))
    with pytest.raises(ValueError, match="native"):
        pool_layer(np.zeros((0, 2), dtype=np.float32), mask)


def test_manifest_integrity(tmp_path, monkeypatch):
    artifact = tmp_path / "readout.safetensors"
    artifact.write_bytes(b"synthetic test artifact")
    manifest = {
        "format_version": 1,
        "encoder": {"repo": ENCODER_REPO, "revision": ENCODER_REVISION},
        "lanes": {
            "turn_end": {
                "files": {
                    "turn_end/readout.safetensors": {
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()
                    }
                }
            }
        },
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(manifest))
    revisions = []

    def download(repo, name, revision):
        revisions.append(revision)
        return str(config if name == "config.json" else artifact)

    monkeypatch.setattr("marimba.weights.hf_hub_download", download)
    result, paths = load_manifest("charcot-ai/marimba", None)
    assert result["resolved_revision"] == DEFAULT_MODEL_REVISION
    assert revisions == [DEFAULT_MODEL_REVISION, DEFAULT_MODEL_REVISION]
    assert paths["turn_end/readout.safetensors"] == artifact
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        load_manifest("charcot-ai/marimba", None)
    manifest["encoder"]["revision"] = "wrong"
    config.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="encoder revision"):
        load_manifest("charcot-ai/marimba", None)


def test_from_pretrained_dispatch(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("marimba.pretrained.load_pretrained", lambda *args: sentinel)
    assert Marimba.from_pretrained() is sentinel


def test_onnx_rejected():
    with pytest.raises(ValueError, match="torch"):
        Marimba.from_pretrained(backend="onnx")


def test_core_import_is_lazy():
    import subprocess

    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sys, marimba; assert 'torch' not in sys.modules; "
            "assert 'qwen_asr' not in sys.modules",
        ],
        text=True,
    )
    assert output == ""


@pytest.mark.real_model
def test_real_model_one_tower_pass():
    model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
    calls = []
    hook = model._engine.thinker.audio_tower.register_forward_hook(lambda *args: calls.append(1))
    questions = [
        Question.score("Has the speaker finished?"),
        Question.choice("What language?", ["en", "pt"]),
        Question.noul('Did someone say "hello"?'),
        Question.noul('Did someone say "world"?'),
    ]
    try:
        decisions = model(np.zeros(16000, dtype=np.float32), questions, 16000)
    finally:
        hook.remove()
    assert calls == [1]
    assert len(decisions) == 4
    assert all(0 <= d.probability <= 1 for d in decisions)
