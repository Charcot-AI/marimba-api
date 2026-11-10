"""Synthetic diarization routing, features and shared-pass decisions."""

import itertools

import numpy as np
import pytest

from marimba import Question, WindowSnapshot
from marimba._diarization import (
    TASKS,
    DiarizationReadout,
    build_same_speaker_features,
    frame_centers_seconds,
    pool_interval,
    task_features,
)
from marimba.pretrained import PretrainedMarimba
from marimba.questions import UnsupportedQuestion, matches_template
from marimba.runtime import RuntimeInfo
from marimba.testing import FakeMarimba
from marimba.windows import EncodedWindow


def weights(task):
    if task == "speaker_change":
        return {
            "layer_0_weight": np.zeros((3584, 256), np.float32),
            "layer_0_bias": np.zeros(256, np.float32),
            "layer_1_weight": np.zeros((256, 1), np.float32),
            "layer_1_bias": np.zeros(1, np.float32),
            "platt_coefficient": np.ones(1, np.float32),
            "platt_intercept": np.ones(1, np.float32),
        }
    result = {
        "scaler_mean": np.zeros(1792, np.float32),
        "scaler_scale": np.ones(1792, np.float32),
        "coef": np.zeros((4 if task == "speaker_count" else 1, 1792), np.float32),
        "intercept": np.arange(4, dtype=np.float32)
        if task == "speaker_count"
        else np.ones(1, np.float32),
    }
    result.update(
        {"temperature": np.ones(1, np.float32)}
        if task == "speaker_count"
        else {
            "platt_coefficient": np.ones(1, np.float32),
            "platt_intercept": np.zeros(1, np.float32),
        }
    )
    return result


def questions():
    return [
        Question.score("Has the speaker finished?"),
        Question.noul("Is this the same speaker?"),
        Question.score("Did the speaker change?"),
        Question.choice("How many speakers?", ["3+", "1", "0", "2"]),
    ]


class Engine:
    def __init__(self):
        self.calls = []
        self.readouts = {t: DiarizationReadout(t, weights(t)) for t in TASKS}

    def encode(self, window):
        self.calls.append("encode")
        return {
            "features": {12: np.ones((104, 896), np.float32)},
            "audio_features": np.zeros((104, 2), np.float32),
        }

    def turn_end(self, features, mask):
        self.calls.append("turn")
        return 0.6, 0.7

    def diarization(self, features, tasks):
        self.calls.append("diarization")
        if 12 not in features:
            raise RuntimeError("Encoded window lacks layer 12 for diarization")
        return {t: self.readouts[t].predict(task_features(t, features[12])[None])[0] for t in tasks}


@pytest.fixture
def model():
    return PretrainedMarimba(
        RuntimeInfo("fake", "cpu", "float32"),
        Engine(),
        {
            "resolved_revision": "a" * 40,
            "lanes": {
                **{
                    t: {
                        "status": "validated",
                        "layer": 12,
                        "files": {f"diarization/{t}.safetensors": {}},
                    }
                    for t in (*TASKS, "turn_end")
                },
                "term": {"calibration": "none"},
            },
        },
    )


@pytest.mark.parametrize(
    "text,task",
    [
        ("Is this the same speaker?", "same_speaker"),
        ("Are these the same speaker?", "same_speaker"),
        (" DID THE SPEAKER CHANGE?! ", "speaker_change"),
        ("Has the speaker changed?", "speaker_change"),
        ("Is this a different speaker?", "speaker_change"),
    ],
)
def test_binary_routing(text, task):
    for kind in (Question.noul, Question.score):
        assert kind(text).task == task
    other = "speaker_change" if task == "same_speaker" else "same_speaker"
    assert not matches_template(other, text)
    with pytest.raises(UnsupportedQuestion):
        Question.choice(text, ["yes", "no"])


@pytest.mark.parametrize("options", list(itertools.permutations(("0", "1", "2", "3+"))))
def test_count_option_order(model, options):
    q = Question.choice("How many speakers are there?", options)
    d = model(np.ones(128000, np.float32), [q], 16000)[0]
    assert d.question is q
    assert list(d.probabilities) == list(options)
    assert d.value == "3+"
    assert d.probability == d.probabilities[d.value]
    assert sum(d.probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "options",
    [[], ["0", "1", "2"], ["0", "1", "2", "3"], ["0", "1", "2", "3+", "4"], ["0", "1", "2", "2"]],
)
def test_invalid_count_options(options):
    with pytest.raises(UnsupportedQuestion):
        Question.choice("How many speakers?", options)


@pytest.mark.parametrize("kind", [Question.noul, Question.score])
def test_count_requires_choice(kind):
    with pytest.raises(UnsupportedQuestion):
        kind("Custom count", task="speaker_count")


def test_shared_encoder_and_binary_conventions(model):
    qs = questions() + [
        Question.score("Is this the same speaker?"),
        Question.noul("Did the speaker change?", threshold=0.9),
    ]
    ds = model(np.ones(128000, np.float32), qs, 16000)
    assert model._engine.calls == ["encode", "diarization", "turn"]
    assert ds[1].value is True
    assert ds[2].value == ds[2].probability == ds[2].score
    assert ds[4].value == ds[1].probability
    assert ds[5].value is False
    assert all(d.question is q for d, q in zip(ds, qs, strict=True))
    assert model(np.ones(128000, np.float32), [], 16000) == []


@pytest.mark.parametrize("factory", [lambda: FakeMarimba(), lambda: None])
def test_partial_windows(model, factory):
    model = factory() or model
    qs = questions()[1:]
    ds = model(np.ones(16000, np.float32), qs, 16000)
    assert all(d.value is d.probability is d.probabilities is d.score is None for d in ds)
    assert all(d.status != "unavailable" for d in ds)
    stream = model.stream(qs)
    updates = stream.push(np.ones(4000, np.float32))
    assert updates and all(d.value is None for u in updates for d in u.decisions)
    if hasattr(model, "_engine"):
        assert "diarization" not in model._engine.calls


def test_batch_snapshot_and_missing_features(model):
    qs = questions()
    model.batch([{"array": np.ones(128000, np.float32), "sampling_rate": 16000}] * 2, qs)
    assert model._engine.calls.count("encode") == 2
    snapshot = WindowSnapshot(np.ones(128000, np.float32), np.ones(128000, bool), 0, 8)
    encoded = model.encode(snapshot)
    model.decide_encoded(encoded, qs)
    assert model._engine.calls.count("encode") == 3
    with pytest.raises(RuntimeError, match="layer 12"):
        model.decide_encoded(EncodedWindow(snapshot), qs)


def test_features_and_errors():
    frames = np.arange(8, dtype=np.float16).reshape(4, 2)
    centers = frame_centers_seconds(4)
    np.testing.assert_array_equal(centers, [1, 3, 5, 7])
    np.testing.assert_array_equal(pool_interval(frames, centers, (0, 4)), [1, 2])
    assert task_features("same_speaker", frames).shape == (4,)
    assert task_features("speaker_change", frames).shape == (8,)
    np.testing.assert_array_equal(task_features("speaker_count", frames)[:2], [3, 4])
    for bad in (np.empty((0, 2)), np.ones(2), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            task_features("speaker_count", bad)
    with pytest.raises(ValueError):
        pool_interval(frames, centers, (0, 0))
    with pytest.raises(ValueError):
        frame_centers_seconds(0)
    with pytest.raises(ValueError):
        build_same_speaker_features(np.zeros(2), np.ones(2))


@pytest.mark.parametrize("task", TASKS)
def test_readout_empty_and_errors(task):
    readout = DiarizationReadout(task, weights(task))
    dim = 3584 if task == "speaker_change" else 1792
    assert readout.predict(np.empty((0, dim))).shape == (0, 4 if task == "speaker_count" else 2)
    for bad in (np.zeros((1, 3)), np.full((1, dim), np.nan)):
        with pytest.raises(ValueError):
            readout.predict(bad)
