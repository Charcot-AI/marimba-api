"""Cog runner parsing and lifecycle tests without Cog, CUDA, or model weights."""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marimba import Decision, Question

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    cog = SimpleNamespace(
        BaseRunner=object, Input=lambda **kwargs: kwargs.get("default"), Path=Path
    )
    monkeypatch.setitem(sys.modules, "cog", cog)
    spec = importlib.util.spec_from_file_location("cog_predictor", ROOT / "cog/predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_questions_cover_every_kind(runner):
    questions = runner.parse_questions(
        inspect.signature(runner.Predictor.run).parameters["questions"].default
    )
    assert {question.kind for question in questions} == {"noul", "score", "choice"}
    assert {question.task for question in questions} == {
        "voice_activity",
        "turn_end",
        "language",
        "term",
        "speaker_count",
    }


def test_parse_questions_accepts_every_supported_field(runner):
    raw = json.dumps(
        [
            {"text": "Is someone speaking?", "kind": "noul", "seconds": 2, "threshold": 0.4},
            {"text": "Did the speaker change?", "kind": "noul", "task": "speaker_change"},
            {"text": "Is this the same speaker?", "kind": "noul"},
            {"text": "Is there any personal information?", "kind": "noul"},
        ]
    )
    questions = runner.parse_questions(raw)
    assert [question.task for question in questions] == [
        "voice_activity",
        "speaker_change",
        "same_speaker",
        "pii",
    ]
    assert questions[0].threshold == 0.4


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "{}",
        "[]",
        json.dumps([{"text": "Is someone speaking?"}]),
        json.dumps([{"text": "Is someone speaking?", "kind": "noul", "extra": 1}]),
        json.dumps([{"text": "What language?", "kind": "choice"}]),
        json.dumps([{"text": "Is someone speaking?", "kind": "noul"}] * 33),
    ],
)
def test_parse_questions_rejects_invalid_payloads(runner, raw):
    with pytest.raises((ValueError, TypeError)):
        runner.parse_questions(raw)


def test_runner_loads_once_and_returns_decision_json(runner, monkeypatch):
    question = Question.noul("Is someone speaking?")
    model = Mock(revision="a" * 40)
    model.return_value = [
        Decision(question, "voice_activity", "noul", True, probability=0.9, status="ok")
    ]
    load = Mock(return_value=model)
    monkeypatch.setattr(runner.Marimba, "from_pretrained", load)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    predictor = runner.Predictor()
    predictor.setup()
    output = predictor.run(
        Path("speech.wav"), json.dumps([{"text": question.text, "kind": question.kind}])
    )
    load.assert_called_once_with(backend="torch")
    assert model.call_args.args[0] == "speech.wav"
    assert output["model_revision"] == "a" * 40
    assert output["decisions"][0]["value"] is True
    json.dumps(output)


def test_requirements_match_package_pins():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    expected = project["dependencies"] + project["optional-dependencies"]["torch"]
    lines = (ROOT / "cog/requirements.txt").read_text().split()

    def name(spec: str) -> str:
        return re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].lower().replace("-", "_")

    pins = {name(line): line for line in lines}
    # The Qwen3-ASR processor imports protobuf, which the torch extra does not pull in.
    assert pins.pop("protobuf") == "protobuf>=4"
    assert pins == {name(spec): spec for spec in expected}


def test_build_downloads_every_pinned_hub_revision():
    build = (ROOT / "cog/cog.yaml").read_text()
    pins = []
    for source in ("weights.py", "_head.py"):
        text = (ROOT / "src/marimba" / source).read_text()
        pins += re.findall(r'^(\w+)_REPO = "(.+)"\n\1_REVISION = "([0-9a-f]{40})"', text, re.M)
    assert len(pins) == 3
    for _, repo, revision in pins:
        assert f"snapshot_download('{repo}',\n        revision='{revision}'" in build
