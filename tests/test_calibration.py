import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from marimba import Calibration, Question
from marimba.calibration import fit_platt
from marimba.testing import FakeMarimba


def test_recover_platt_and_preserve_ranking():
    rng = np.random.default_rng(443)
    logits = rng.uniform(-3, 3, 30000)
    probabilities = 1 / (1 + np.exp(-logits))
    labels = rng.binomial(1, 1 / (1 + np.exp(-(1.8 * logits - 0.4))))
    a, b = fit_platt(probabilities, labels)
    assert a == pytest.approx(1.8, abs=0.08)
    assert b == pytest.approx(-0.4, abs=0.05)
    calibration = Calibration(
        "question", "noul", "platt", a, b, 30000, int(sum(labels)), "revision"
    )
    grid = np.linspace(0.01, 0.99, 100)
    assert np.all(np.diff([calibration.apply(p) for p in grid]) > 0)
    assert Calibration.from_dict(json.loads(json.dumps(calibration.to_dict()))) == calibration
    with pytest.raises(FrozenInstanceError):
        calibration.a = 2


def test_calibrate_attach_save_load_clear_and_refit(tmp_path):
    model = FakeMarimba()
    q = Question.noul("Is someone speaking?")
    audio = [
        {"array": np.full(1600, value, dtype=np.float32), "sampling_rate": 16000}
        for value in [0.01, 0.1, 0.2, 0.3]
    ]
    raw = model.batch(audio, [q])
    with pytest.warns(UserWarning, match="fewer than 100"):
        cal = model.calibrate(audio, [0, 0, 1, 1], q)
    assert cal.n == 4 and cal.n_positive == 2
    assert cal.created_with is None
    decisions = model.batch(audio, [q])
    assert all(row[0].calibrated for row in decisions)
    assert decisions[0][0].probability == cal.apply(raw[0][0].probability)
    path = tmp_path / "calibrations.json"
    model.save_calibrations(path)
    fresh = FakeMarimba()
    fresh.load_calibrations(path)
    assert fresh(audio[0], [q])[0].probability == decisions[0][0].probability
    with pytest.warns(UserWarning):
        second = model.calibrate(audio, [0, 0, 1, 1], q)
    assert (second.a, second.b) == (cal.a, cal.b)
    stream = fresh.stream([q])
    stream.ingest(np.full(1600, 0.1, dtype=np.float32))
    encoded = fresh.encode(stream.snapshot())
    assert fresh.decide_encoded(encoded, [q])[0].calibrated
    assert stream.decide().decisions[0].calibrated
    assert stream.push(np.zeros(4000, dtype=np.float32))[0].decisions[0].calibrated
    fresh.clear_calibration(q)
    assert not fresh(audio[0], [q])[0].calibrated


def test_score_and_dataset_column():
    datasets = pytest.importorskip("datasets")
    q = Question.score("Is someone speaking?")
    audio = [{"array": [value] * 100, "sampling_rate": 16000} for value in [0.01, 0.1, 0.2, 0.3]]
    column = datasets.Dataset.from_dict({"audio": audio})["audio"]
    model = FakeMarimba()
    with pytest.warns(UserWarning):
        model.calibrate(column, [False, False, True, True], q, batch_size=2)
    decision = model(column[0], [q])[0]
    assert decision.calibrated and decision.value == decision.probability


def test_calibration_input_and_persistence_errors(tmp_path):
    model = FakeMarimba()
    q = Question.noul("Is someone speaking?")
    with pytest.raises(ValueError):
        model.calibrate([], [], Question.choice("What language?", ["en", "pt"]))
    for audio, labels in [([], []), ([0], [0]), ([0, 1], [0, 2]), ([0, 1], [1]), ([0, 1], [1, 1])]:
        with pytest.raises(ValueError):
            model.calibrate(audio, labels, q)
    with pytest.raises(ValueError):
        model.calibrate([0, 1], [0, 1], q, method="unknown")
    for p, y in [([float("nan"), 0.5], [0, 1]), ([-0.1, 0.5], [0, 1]), ([0.2, 0.3], [1, 1])]:
        with pytest.raises(ValueError):
            fit_platt(p, y)
    path = tmp_path / "bad.json"
    for data in [{}, {"format_version": 1}, {"format_version": 1, "calibrations": [{}]}]:
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            model.load_calibrations(path)
