"""Synthetic Space checks and executed-notebook release hygiene."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marimba.testing import FakeMarimba

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_module():
    pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_decisions(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    decisions = app_module.decide(str(audio), app_module.DEFAULT_QUESTIONS)
    assert [d["task"] for d in decisions] == ["turn_end", "voice_activity"]
    json.dumps(decisions)
    assert len(app_module.build_app().config["dependencies"]) == 1


@pytest.mark.parametrize("raw", ["", "{}", "[]", "[1]", "[{}]", '[{"text":"unknown"}]'])
def test_space_invalid_questions(app_module, raw):
    with pytest.raises((TypeError, ValueError)):
        app_module.parse_questions(raw)


def test_space_same_speaker_partial_window(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    decisions = app_module.decide(
        str(audio), json.dumps([{"text": "Is this the same speaker?", "kind": "noul"}])
    )
    assert decisions[0]["status"] == "experimental"
    assert decisions[0]["value"] is decisions[0]["probability"] is decisions[0]["score"] is None


def test_space_empty_audio_and_sanitized_failure(app_module, monkeypatch):
    with pytest.raises(app_module.gr.Error, match="Upload"):
        app_module.decide(None, app_module.DEFAULT_QUESTIONS)
    with pytest.raises(app_module.gr.Error, match="Invalid questions"):
        app_module.decide("unused.wav", "[]")

    def fail():
        raise RuntimeError("Sensitive detail")

    monkeypatch.setattr(app_module, "load_model", fail)
    with pytest.raises(app_module.gr.Error, match="Inference failed") as error:
        app_module.decide("unused.wav", app_module.DEFAULT_QUESTIONS)
    assert "Sensitive detail" not in str(error.value)


def test_executed_notebooks_are_small_and_text_only():
    paths = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
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

    colab_path = ROOT / "notebooks" / "colab_quickstart.ipynb"
    assert colab_path.is_file()
    colab = json.loads(colab_path.read_text())
    colab_code = [cell for cell in colab["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] > 0 for cell in colab_code)
    assert colab_path.stat().st_size < 30000
    colab_text = json.dumps(colab)
    assert "/home/" not in colab_text
    assert "hf-internal-testing/librispeech_asr_dummy" in colab_text
    assert "sanchit-gandhi/librispeech_asr_dummy" in colab_text
    assert "Synthetic audio:" not in colab_text
    assert "[PENDING]" in colab_text


def test_documentation_relative_links_resolve():
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "notebooks" / "README.md"]
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
