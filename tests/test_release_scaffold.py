"""Synthetic Space checks and executed-notebook release hygiene."""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marimba.testing import FakeMarimba

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_module(monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=lambda **_: lambda fn: fn))
    spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_decisions(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    decisions = app_module.decide(str(audio), ["Speech", "Turn ended"], "")
    assert [d["task"] for d in decisions] == ["voice_activity", "turn_end"]
    json.dumps(decisions)
    app = app_module.build_app()
    assert any(item["api_name"] == "decide" for item in app.config["dependencies"])
    audio_component = next(item for item in app.config["components"] if item["type"] == "audio")
    assert audio_component["props"]["sources"] == ["upload", "microphone"]


@pytest.mark.parametrize("selected", [[], ["Unknown"], ["Speech", "Speech"], [None]])
def test_space_invalid_questions(app_module, selected):
    with pytest.raises((TypeError, ValueError)):
        app_module.make_questions(selected, "marimba")


@pytest.mark.parametrize("term", ["", "   ", 'a"b', "a\nb", "a" * 101, "..."])
def test_space_invalid_terms(app_module, term):
    with pytest.raises(ValueError):
        app_module.make_questions(["Term"], term)


def test_space_typed_questions(app_module):
    questions = app_module.make_questions(app_module.QUESTION_TYPES, "marimba")
    assert [q.task for q in questions] == [
        "voice_activity",
        "turn_end",
        "language",
        "term",
        "speaker_change",
        "speaker_count",
    ]
    assert questions[3].term == "marimba"
    assert questions[-1].options == ("0", "1", "2", "3+")


def test_space_model_loads_once(app_module, monkeypatch):
    calls = []
    model = FakeMarimba()

    def pretrained(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(app_module.Marimba, "from_pretrained", pretrained)
    monkeypatch.setenv("MARIMBA_DEVICE", "cuda")
    assert app_module.load_model() is app_module.load_model() is model
    assert calls == [{"backend": "torch", "device": "cuda", "dtype": "float32"}]


def test_space_deployment_bundle():
    spec = importlib.util.spec_from_file_location("space_deploy", ROOT / "space" / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.deployment_files()
    assert files["requirements.txt"] == (ROOT / "space" / "requirements.txt").read_bytes()
    assert files["README.md"] == ROOT / "space" / "README.md"
    assert "src/marimba/lexical.py" in files
    assert "LICENSE" in files
    assert len([name for name in files if name.endswith(".wav")]) == 3
    assert not any(name.startswith((".git", "demo/", "tests/")) for name in files)
    assert not any("__pycache__" in name or ".env" in name for name in files)


def test_space_examples_are_eight_second_pcm():
    for path in (ROOT / "space" / "examples").glob("*.wav"):
        info = sf.info(path)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.frames == 128000
        assert info.subtype == "PCM_16"


def test_space_diarization_partial_window(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    decisions = app_module.decide(str(audio), ["Speaker change", "People"], "")
    for decision in decisions:
        assert decision["status"] == "experimental"
        assert decision["value"] is decision["probability"] is decision["score"] is None


def test_space_empty_audio_and_sanitized_failure(app_module, monkeypatch):
    with pytest.raises(app_module.gr.Error, match="Upload"):
        app_module.decide(None, ["Speech"], "")
    with pytest.raises(app_module.gr.Error, match="Select"):
        app_module.decide("unused.wav", [], "")

    def fail():
        raise RuntimeError("Sensitive detail")

    monkeypatch.setattr(app_module, "load_model", fail)
    with pytest.raises(app_module.gr.Error, match="Inference failed") as error:
        app_module.decide("unused.wav", ["Speech"], "")
    assert "Sensitive detail" not in str(error.value)


def test_executed_notebooks_are_small_and_text_only():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = json.loads(path.read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert all(c["execution_count"] > 0 for c in cells)
        assert path.stat().st_size < 30000
        assert all(
            o["output_type"] == "stream" and o["name"] == "stdout"
            for cell in cells
            for o in cell["outputs"]
        )
        text = json.dumps(notebook)
        assert "70bb2e84b976b7e960aa89f1c648e09c59f894dd" in text
        assert "FakeMarimba" not in text
        assert "/home/" not in text


def test_documentation_relative_links_resolve():
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "notebooks" / "README.md"]
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
