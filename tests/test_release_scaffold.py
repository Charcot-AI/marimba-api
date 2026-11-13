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
    assert audio_component["props"]["sources"] == ["microphone", "upload"]
    assert audio_component["props"]["label"] == "Record or upload audio"


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
    assert questions[0].seconds == 2.0
    assert questions[0].resolved_threshold == 0.225
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


def test_space_demo_questions_reuse_cards_with_only_user_term_changed(app_module):
    from marimba_demo.cards import available_cards, questions_for

    from marimba import Question

    model = FakeMarimba()
    expected = questions_for(available_cards(model), Question)
    actual = app_module.demo_questions(model, "  piano  ")
    assert actual[:-1] == expected[:-1]
    assert actual[-1].text == 'Was "piano" spoken?'
    assert actual[-1].task == "term"
    assert actual[-1].term == "piano"
    cards = app_module.demo_cards(model, "piano")
    assert cards[-1].title == '"piano" spoken'
    assert app_module.present([], cards)[-2].startswith('### "piano" spoken')
    assert actual[0].kind == "noul"
    assert actual[1].seconds == 2.0
    assert actual[-2].options == ("en", "pt", "es")


@pytest.mark.parametrize("value", [None, "0", "true", "1"])
def test_space_live_mode_switch(app_module, monkeypatch, value):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    if value is None:
        monkeypatch.delenv("MARIMBA_LIVE", raising=False)
    else:
        monkeypatch.setenv("MARIMBA_LIVE", value)
    app = app_module.build_app()
    microphones = [
        c for c in app.config["components"] if c["type"] == "audio" and c["props"].get("streaming")
    ]
    assert bool(microphones) == (value == "1")
    assert app_module.live_enabled() == (value == "1")
    assert {"decide", "replay"} <= {d["api_name"] for d in app.config["dependencies"]}
    if microphones:
        assert microphones[0]["props"]["sources"] == ["microphone"]
        assert any(d["stream_every"] == 0.25 for d in app.config["dependencies"])


def test_space_replay_api_and_duration_limit(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.zeros(16000, np.float32), 16000)
    events = app_module.run_replay(str(path), "marimba")
    assert len(events) == 12
    assert all(e["coalesced_hops"] == 0 for e in events)
    assert events[0]["replay"] == {
        "input_seconds": 1.0,
        "trailing_silence_s": 2.0,
        "frame_padding_s": 0.0,
    }
    sf.write(path, np.zeros(480001, np.float32), 16000)
    with pytest.raises(app_module.gr.Error, match="30 seconds"):
        app_module.run_replay(str(path), "marimba")
    with pytest.raises(app_module.gr.Error, match="Upload"):
        app_module.run_replay(None, "marimba")
    with pytest.raises(app_module.gr.Error, match="term"):
        app_module.run_replay(str(path), 'bad"term')


def test_space_live_buffers_resamples_and_does_not_pad_at_stop(app_module):
    import asyncio

    from space.live import LiveController

    async def run():
        controller = LiveController(FakeMarimba(), app_module.demo_questions)
        chunk = (48000, np.full((48480, 2), 16384, np.int16))
        state = controller.start(None, "marimba", "first")
        state = await controller.stream(chunk, state, "marimba", "first")
        with pytest.raises(ValueError, match="Busy"):
            await controller.stream(chunk, None, "marimba", "other")
        assert controller.owner is state
        await controller.close(state)
        assert state.run.scheduler.total == 16000
        assert len(state.buffer) == 160
        assert state.closed and controller.owner is None
        assert all(worker.done() for worker in state.run.workers)
        np.testing.assert_allclose(state.run.scheduler.audio[100:-100], 0.5, atol=0.001)

    asyncio.run(run())


def test_space_live_admits_before_audio_and_rejects_another_session(app_module):
    import asyncio

    from space.live import LiveController

    async def run():
        controller = LiveController(FakeMarimba(), app_module.demo_questions)
        first = controller.start(None, "marimba", "first")
        assert controller.start(None, "marimba", "first") is first
        assert first.run.scheduler.total == 0
        with pytest.raises(ValueError, match="Busy"):
            controller.start(None, "marimba", "second")
        await controller.close(first)
        replacement = controller.start(None, "marimba", "second")
        assert controller.owner is replacement
        await controller.close(replacement, aborted=True)

    asyncio.run(run())


def test_space_live_admission_and_stop_do_not_queue_behind_stream(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "load_model", FakeMarimba)
    monkeypatch.setenv("MARIMBA_LIVE", "1")
    app = app_module.build_app()
    callbacks = {block.fn.__name__: block for block in app.fns.values() if block.fn is not None}
    assert callbacks["stream"].concurrency_limit is None
    assert callbacks["start"].queue is False
    assert callbacks["stop"].queue is False


def test_space_live_coalesces_and_disconnect_releases_owner(app_module):
    import asyncio

    from space.live import LiveController

    async def run():
        controller = LiveController(FakeMarimba(), app_module.demo_questions)
        chunk = (16000, np.zeros(16000, np.float32))
        state = controller.start(None, "marimba", "first")
        state = await controller.stream(chunk, state, "marimba", "first")
        await state.run.wait_for_hop(3)
        assert state.run.scheduler.coalesced["encoder"] == 3
        await controller.disconnect("other")
        assert controller.owner is state
        await controller.disconnect("first")
        assert controller.owner is None and state.closed
        for stale in (None, state):
            with pytest.raises(ValueError, match="Start recording"):
                await controller.stream(chunk, stale, "marimba", "first")
            assert controller.owner is None
        replacement = controller.start(None, "marimba", "second")
        with pytest.raises(ValueError, match="Busy"):
            await controller.stream(chunk, state, "marimba", "first")
        assert controller.owner is replacement
        await controller.close(replacement, aborted=True)

    asyncio.run(run())


def test_space_live_invalid_first_chunk_releases_owner(app_module):
    import asyncio

    from space.live import LiveController

    async def run():
        controller = LiveController(FakeMarimba(), app_module.demo_questions)
        state = controller.start(None, "marimba", "first")
        with pytest.raises(ValueError, match="sample rate"):
            await controller.stream((0, np.zeros(4000, np.float32)), state, "marimba", "first")
        assert controller.owner is None

    asyncio.run(run())


def test_space_replay_presentation(app_module):
    from marimba_demo.cards import available_cards

    events = [
        {
            "decisions": [
                {
                    "task": "voice_activity",
                    "value": True,
                    "probability": 0.9,
                    "window_start_s": 0.0,
                    "window_end_s": 2.0,
                },
                {
                    "task": "term",
                    "value": True,
                    "probability": 0.6,
                    "window_start_s": 0.5,
                    "window_end_s": 2.0,
                },
            ]
        }
    ]
    values = app_module.present(events, available_cards(FakeMarimba()))
    assert "0.9000" in values[1]
    assert "raw N-best fraction 0.6000" in values[-2]
    assert "0.50–2.00 s" in values[-2]
    assert values[-1].to_dict("records") == [
        {"window_end_s": 2.0, "probability": 0.9, "card": "Speech"}
    ]


def test_space_user_term_card_title_is_literal_markdown(app_module):
    cards = app_module.demo_cards(FakeMarimba(), "piano [example]")
    assert app_module.card_values(cards, {}, [])[-1].startswith('### "piano \\[example\\]" spoken')


def test_space_deployment_bundle():
    spec = importlib.util.spec_from_file_location("space_deploy", ROOT / "space" / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.deployment_files()
    assert files["requirements.txt"] == (ROOT / "space" / "requirements.txt").read_bytes()
    assert files["README.md"] == ROOT / "space" / "README.md"
    assert "src/marimba/lexical.py" in files
    assert "LICENSE" in files
    assert len([name for name in files if name.endswith(".wav")]) == 8
    assert "space/examples/manifest.json" in files
    assert {name for name in files if name.startswith("demo/")} == {
        f"demo/marimba_demo/{name}.py"
        for name in ("__init__", "cards", "protocol", "scheduler", "speech", "replay")
    }
    assert {"space/live.py", "space/presentation.py"} <= files.keys()
    assert not any(name.startswith((".git", "tests/")) for name in files)
    assert not any("__pycache__" in name or ".env" in name for name in files)


@pytest.mark.parametrize("hardware", ["zero-a10g", "l4x1"])
def test_space_deployment_live_variable_follows_hardware(monkeypatch, hardware):
    calls = []

    class API:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, *args, **kwargs):
            assert kwargs["private"] is True

        def space_info(self, *args):
            return types.SimpleNamespace(
                private=True, runtime=types.SimpleNamespace(requested_hardware=hardware)
            )

        def request_space_hardware(self, *args, **kwargs):
            assert kwargs["sleep_time"] == (300 if hardware == "l4x1" else None)

        def add_space_secret(self, *args, **kwargs):
            pass

        def get_space_variables(self, *args):
            return {"MARIMBA_LIVE": "1"}

        def add_space_variable(self, *args, **kwargs):
            calls.append(("set", kwargs))

        def delete_space_variable(self, *args, **kwargs):
            calls.append(("delete", kwargs))

        def list_repo_files(self, *args, **kwargs):
            return []

        def create_commit(self, *args, **kwargs):
            return types.SimpleNamespace(oid="synthetic-commit")

    spec = importlib.util.spec_from_file_location("space_deploy", ROOT / "space" / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "HfApi", API)
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    monkeypatch.setattr(
        sys, "argv", ["deploy.py", "--repo-id", "example/test", "--hardware", hardware]
    )
    module.main()
    assert calls == [
        ("set", {"key": "MARIMBA_LIVE", "value": "1"})
        if hardware == "l4x1"
        else ("delete", {"key": "MARIMBA_LIVE"})
    ]


def test_space_deployment_stops_before_updating_a_public_space(monkeypatch):
    class API:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, *args, **kwargs):
            pass

        def space_info(self, *args):
            return types.SimpleNamespace(private=False)

    spec = importlib.util.spec_from_file_location("space_deploy", ROOT / "space" / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "HfApi", API)
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")
    monkeypatch.setattr(sys, "argv", ["deploy.py", "--repo-id", "example/test"])
    with pytest.raises(RuntimeError, match="Refusing to update a public Space"):
        module.main()


def test_space_examples_are_eight_second_pcm():
    import hashlib

    examples = ROOT / "space" / "examples"
    manifest = json.loads((examples / "manifest.json").read_text())
    assert len(manifest["scenes"]) == 8
    for scene in manifest["scenes"]:
        assert scene["end_sample"] == scene["segments"][-1]["end_sample"] + 4800
        assert scene["post_speech_samples"] == 4800
        path = examples / f"{scene['id']}.wav"
        info = sf.info(path)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.frames == 128000
        assert info.subtype == "PCM_16"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == scene["sha256"]


def test_scene_cuts_preserve_samples_and_pad_only_at_start():
    spec = importlib.util.spec_from_file_location(
        "space_examples", ROOT / "space/generate_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = np.arange(140000, dtype=np.int16)
    short = module.cut_scene(audio, 10, 20)
    assert len(short) == 128000
    assert np.array_equal(short[-10:], audio[10:20])
    assert not short[:-10].any()
    assert np.array_equal(module.cut_scene(audio, 0, 140000), audio[-128000:])
    with pytest.raises(ValueError):
        module.cut_scene(audio, 20, 10)


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
