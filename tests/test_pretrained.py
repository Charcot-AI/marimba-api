import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from marimba import Calibration, Marimba, Question, RuntimeInfo, UnsupportedQuestion, WindowSnapshot
from marimba.audio import right_aligned_window
from marimba.lexical import exact_fraction, normalize
from marimba.pretrained import PretrainedMarimba
from marimba.readout import pool_layer, predict_readout
from marimba.weights import DEFAULT_MODEL_REVISION, ENCODER_REPO, ENCODER_REVISION, load_manifest


class StubEngine:
    def __init__(self):
        self.calls = []
        self.head_texts = []

    def encode(self, window):
        self.calls.append("encode")
        return {
            "features": {
                9: np.zeros((100, 2), dtype=np.float32),
                18: np.zeros((100, 2), dtype=np.float32),
            },
            "audio_features": np.zeros((100, 2), dtype=np.float32),
        }

    def turn_end(self, features, mask):
        self.calls.append("turn_end")
        return 0.6, 0.7

    def hypotheses(self, encoded, prefix=None):
        self.calls.append("hypotheses")
        return ["Hello world"] * 4 + ["World"]

    def transcript(self, encoded, prefix=None):
        self.calls.append("transcript")
        return "Greedy transcript"

    def language_scores(self, encoded, options, prefix=None):
        self.calls.append("language")
        return {option: -float(index + 1) for index, option in enumerate(options)}

    def head_score(self, features, mask, text):
        self.calls.append("head")
        self.head_texts.append(text)
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
                "status": "validated",
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
    assert model._engine.calls == ["encode", "turn_end", "encode", "language", "hypotheses"]
    assert decisions[0].probability == 0.7
    assert decisions[0].score == 0.6
    assert decisions[1].value == "en"
    assert sum(decisions[1].probabilities.values()) == pytest.approx(1)
    assert [decision.probability for decision in decisions[2:]] == [0.8, 0.0]
    assert all(json.loads(json.dumps(d.to_dict())) == d.to_dict() for d in decisions)
    assert model.capabilities["term"].status == "validated"
    assert "Uncalibrated" in model.capabilities["term"].evaluation


@pytest.mark.parametrize(
    "probability,threshold,expected",
    [
        (0.49, None, False),
        (0.5, None, True),
        (0.9, None, True),
        (0.7, 0.8, False),
    ],
)
def test_turn_end_noul_thresholds_platt_probability_not_raw_score(
    model, probability, threshold, expected
):
    model._engine.turn_end = lambda features, mask: (40.0, probability)
    question = Question.noul("Has the speaker finished?", threshold=threshold)
    binary, score = model(np.ones(16000), [question, Question.score(question.text)], 16000)
    assert "noul" in model.capabilities["turn_end"].kinds
    assert binary.kind == "noul"
    assert binary.value is expected
    assert binary.probability == probability
    assert binary.score == 40.0
    assert binary.threshold == (0.5 if threshold is None else threshold)
    assert score.value == probability


@pytest.mark.parametrize("route", ["call", "snapshot", "encoded"])
def test_partial_decoder_audio_is_left_aligned_on_every_entry_point(model, route):
    windows = []
    masks = []
    original_encode = model._engine.encode
    original_turn = model._engine.turn_end

    def encode(window):
        windows.append(window.copy())
        return original_encode(window)

    def turn(features, mask):
        masks.append(mask.copy())
        return original_turn(features, mask)

    model._engine.encode, model._engine.turn_end = encode, turn
    samples = np.ones(16000, np.float32)
    samples[400:800] = 0
    window, mask = right_aligned_window(samples)
    snapshot = WindowSnapshot(window, mask, 10, 11)
    decoder = Question.noul('Was "hello" spoken?')
    questions = [Question.score("Has the speaker finished?"), decoder]
    if route == "call":
        decisions = model(samples, questions, 16000)
    elif route == "snapshot":
        decisions = model.decide_snapshot(snapshot, [decoder])
    else:
        decisions = model.decide_encoded(model.encode(snapshot), questions)
    np.testing.assert_array_equal(windows[-1][:16000], samples)
    assert not windows[-1][16000:].any()
    if route != "snapshot":
        np.testing.assert_array_equal(windows[0], window)
        np.testing.assert_array_equal(masks[0], mask)
    assert decisions[-1].window_end_s == (1 if route == "call" else 11)


def test_language_names_other_dispatch_and_calibration(model):
    from marimba.language import LANGUAGES

    requested = []

    def scores(encoded, options, prefix=None):
        requested.extend(options)
        return dict.fromkeys(options, 0.0)

    model._engine.language_scores = scores
    named = Question.choice("What language?", ["English", "Portuguese", "Spanish", "Other"])
    iso = Question.choice("Which language?", ["en", "pt"])
    audio = np.zeros(128000, dtype=np.float32)
    decisions = model(audio, [named, iso], 16000)
    assert set(requested) == set(LANGUAGES) and len(requested) == 30
    assert decisions[0].probabilities["Other"] == pytest.approx(0.9)
    assert decisions[1].probabilities == {"en": 0.5, "pt": 0.5}
    q = Question.score("Has the speaker finished?")
    inputs = [{"array": audio, "sampling_rate": 16000}] * 100
    cal = model.calibrate(inputs, [0, 1] * 50, q)
    assert cal.created_with == model.revision
    enc = model.encode(WindowSnapshot(audio, np.ones(128000, dtype=bool), 0, 8))
    assert model.decide_encoded(enc, [q])[0].calibrated
    with pytest.raises(ValueError, match="without probabilities"):
        model.calibrate(inputs, [0, 1] * 50, Question.noul("Is someone speaking?"))


def test_unavailable_and_empty(model):
    question = Question.noul("Is someone speaking?")
    decision = model(np.zeros(1, dtype=np.float32), [question], 16000)[0]
    assert decision.status == "unavailable"
    assert decision.value is None and decision.probability is None
    assert model._engine.calls == []
    assert model(np.zeros(1, dtype=np.float32), [], 16000) == []
    with pytest.raises(ValueError, match="nonempty"):
        model(np.array([], dtype=np.float32), [question], 16000)


def test_pii_unavailable_without_detector(model):
    question = Question.noul("Does this audio contain personal information?")
    decision = model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    capability = model.capabilities["pii"]
    assert capability.status == "unavailable"
    assert capability.lane == "decoder" and capability.kinds == ("noul",)
    assert capability.evaluation == "Requires a pii_detector"
    assert decision.status == "unavailable"
    assert (
        decision.value is decision.probability is decision.probabilities is decision.score is None
    )
    assert not decision.calibrated
    assert json.loads(json.dumps(decision.to_dict())) == decision.to_dict()
    assert model._engine.calls == []


def _pii_model(detector):
    engine = StubEngine()
    manifest = {
        "resolved_revision": DEFAULT_MODEL_REVISION,
        "lanes": {
            "term": {"status": "experimental", "calibration": "none"},
        },
    }
    return PretrainedMarimba(
        RuntimeInfo("fake", "cpu", "float32"), engine, manifest, pii_detector=detector
    )


@pytest.mark.parametrize("detector_result", [True, False])
def test_pii_detector_uses_greedy_transcript(detector_result):
    received = []

    def detector(transcript):
        received.append(transcript)
        return detector_result

    model = _pii_model(detector)
    question = Question.noul("Is there PII?")
    model._calibrations[(question.text, question.kind)] = Calibration(
        question.text, question.kind, "platt", 1.0, 0.0, 100, 50, model.revision
    )
    decision = model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    assert received == ["Greedy transcript"]
    assert model._engine.calls == ["encode", "transcript"]
    assert decision.value is detector_result
    assert decision.probability is decision.score is decision.probabilities is None
    assert decision.status == "experimental"
    assert not decision.calibrated
    assert json.loads(json.dumps(decision.to_dict())) == decision.to_dict()
    assert model.capabilities["pii"].evaluation == (
        "Your detector is unevaluated; a private, curated PII pipeline "
        "scored item F1 0.904 on synthetic speech"
    )


def test_term_and_pii_share_greedy_decode():
    received = []

    def detector(transcript):
        received.append(transcript)
        return True

    model = _pii_model(detector)
    questions = [
        Question.noul('Did someone say "hello"?'),
        Question.noul("Does this audio contain personal information?"),
    ]
    decisions = model(np.zeros(16000, dtype=np.float32), questions, 16000)
    assert model._engine.calls == ["encode", "hypotheses"]
    assert received == ["Hello world"]
    assert decisions[0].probability == 0.8
    assert decisions[1].value is True


def test_language_and_pii_share_prefill():
    model = _pii_model(lambda transcript: True)
    model._capabilities["language"] = replace(model.capabilities["language"], status="validated")
    calls = []
    prefix = object()

    def prefill(encoded):
        calls.append("prefill")
        return prefix

    def language_scores(encoded, options, prefix=None):
        calls.append(("language", prefix))
        return {option: 0.0 for option in options}

    def transcript(encoded, prefix=None):
        calls.append(("transcript", prefix))
        return "Greedy transcript"

    model._engine.prefill = prefill
    model._engine.language_scores = language_scores
    model._engine.transcript = transcript
    questions = [Question.choice("What language?", ["en", "pt"]), Question.noul("Is there PII?")]
    model(np.zeros(16000, dtype=np.float32), questions, 16000)
    assert calls == ["prefill", ("language", prefix), ("transcript", prefix)]


def test_pii_supports_noul_only(model):
    question = Question.choice("Does this audio contain personal information?", ["yes", "no"])
    with pytest.raises(UnsupportedQuestion, match="supports kinds"):
        model._validate([question])


def test_shared_speech_head(model):
    model._capabilities["voice_activity"] = replace(
        model.capabilities["voice_activity"], status="validated"
    )
    questions = [
        Question.score("Has the speaker finished?"),
        Question.noul("Is someone speaking?"),
        Question.noul("Is this the same speaker?"),
        Question.score("Is this the same speaker?"),
        Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
    ]
    decisions = model(np.zeros(16000, dtype=np.float32), questions, 16000)
    assert model._engine.calls == ["encode", "turn_end", "head"]
    assert [d.status for d in decisions] == [
        "validated",
        "validated",
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    assert decisions[1].score == 2.0
    assert decisions[1].probability == pytest.approx(1 / (1 + np.exp(-2 / 0.6540604692061299)))
    assert decisions[1].value is True
    assert decisions[1].threshold == 0.225
    assert not decisions[1].calibrated
    for decision in decisions[2:]:
        assert (
            decision.value
            is decision.probability
            is decision.probabilities
            is decision.score
            is None
        )


@pytest.fixture
def vad_model():
    manifest = {
        "resolved_revision": DEFAULT_MODEL_REVISION,
        "lanes": {
            "voice_activity": {"status": "validated", "evaluation": "English prompts"},
            "term": {"calibration": "none"},
        },
    }
    return PretrainedMarimba(RuntimeInfo("fake", "cpu", "float32"), StubEngine(), manifest)


@pytest.mark.parametrize(
    "text",
    ["Is someone speaking?", "Is anyone talking?", "Is there speech?", " IS SOMEONE SPEAKING! "],
)
@pytest.mark.parametrize("task", [None, "voice_activity"])
def test_vad_recognized_prompt(vad_model, text, task):
    question = Question.noul(text, task=task)
    decision = vad_model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    assert vad_model._engine.head_texts == ["Há fala humana neste áudio?"]
    assert decision.question is question
    assert decision.question.text == text
    assert decision.status == "validated"
    evaluation = vad_model.capabilities["voice_activity"].evaluation
    assert "English prompts" not in evaluation
    assert "validated with Portuguese prompts and held-out Arabic, Hindi, Korean and Russian" in (
        evaluation
    )
    assert "temperature-scaled on validation" in evaluation
    assert "default threshold 0.225 from validation" in evaluation


def test_vad_temperature_and_default_threshold(vad_model, monkeypatch):
    monkeypatch.setattr(vad_model._engine, "head_score", lambda features, mask, text: -0.6)
    question = Question.noul("Is someone speaking?")
    decision = vad_model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    expected = 1 / (1 + np.exp(0.6 / 0.6540604692061299))
    assert decision.score == -0.6
    assert decision.probability == pytest.approx(expected)
    assert decision.probability == pytest.approx(0.2855, abs=0.0001)
    assert decision.threshold == 0.225
    assert decision.value is True
    assert not decision.calibrated


@pytest.mark.parametrize(
    "text", ["Custom voice detector wording", "Is someone speaking right now?"]
)
def test_vad_explicit_prompt(vad_model, text):
    question = Question.noul(text, task="voice_activity")
    decision = vad_model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    assert vad_model._engine.head_texts == [text]
    assert decision.question is question


def test_vad_empty_questions_and_invalid_text(vad_model):
    assert vad_model(np.zeros(16000, dtype=np.float32), [], 16000) == []
    assert vad_model._engine.calls == []
    with pytest.raises(ValueError, match="Question text"):
        Question.noul("", task="voice_activity")
    with pytest.raises(UnsupportedQuestion, match="No routing template"):
        Question.noul("Unrecognized voice detector wording")


@pytest.mark.parametrize(
    "question",
    [
        Question.noul("Custom", task="same_speaker"),
        Question.noul("Custom", task="speaker_change"),
        Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
        Question.choice("Custom", ["speech", "noise"], task="audio_class"),
    ],
)
def test_unreleased_lanes_ignore_manifest_without_inference(model, question):
    assert model.capabilities[question.task].status == "unavailable"
    decision = model(np.zeros(16000, dtype=np.float32), [question], 16000)[0]
    assert decision.status == "unavailable"
    assert (
        decision.value is decision.probability is decision.probabilities is decision.score is None
    )
    assert not decision.calibrated
    assert decision.window_start_s == 0 and decision.window_end_s == 1
    assert model._engine.calls == []
    assert json.loads(json.dumps(decision.to_dict())) == decision.to_dict()
    assert model.batch([], [question]) == []
    with pytest.raises(ValueError, match="nonempty"):
        model(np.array([], dtype=np.float32), [question], 16000)
    assert model._engine.calls == []


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


def test_pretrained_split_locks_and_owned_features(model, monkeypatch):
    entered, release = Event(), Event()
    term = Question.noul('Did someone say "hello"?')
    turn = Question.score("Has the speaker finished?")
    snapshot = WindowSnapshot(np.ones(128000, dtype=np.float32), np.ones(128000, dtype=bool), 0, 8)
    encoded = model.encode(snapshot)
    previous = encoded.l18.copy()
    original = model._engine.hypotheses

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(model._engine, "hypotheses", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decoder = pool.submit(model.decide_encoded, encoded, [term])
        try:
            assert entered.wait(5)

            def encoder_work():
                return model.decide_encoded(model.encode(snapshot), [turn])

            assert pool.submit(encoder_work).result(1)[0].probability == 0.7
            np.testing.assert_array_equal(encoded.l18, previous)
            assert not decoder.done()
        finally:
            release.set()
        assert decoder.result(5)[0].probability == 0.8


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
