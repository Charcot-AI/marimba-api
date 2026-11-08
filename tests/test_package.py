import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from marimba import Question, UnsupportedQuestion
from marimba.audio import decode_audio, right_aligned_window
from marimba.cli import main
from marimba.runtime import select_backend
from marimba.testing import FakeMarimba


@pytest.fixture
def sine():
    return np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000).astype(np.float32)


def test_audio_forms(tmp_path, sine):
    buffer = io.BytesIO()
    sf.write(buffer, np.column_stack([sine, sine]), 8000, format="WAV", subtype="FLOAT")
    path = tmp_path / "synthetic.wav"
    path.write_bytes(buffer.getvalue())
    forms = [
        path,
        str(path),
        {"path": str(path), "bytes": None},
        {"path": "missing", "bytes": buffer.getvalue()},
        {"array": sine, "sampling_rate": 8000},
        sine,
    ]
    expected = decode_audio(sine, 8000)
    for form in forms:
        actual = decode_audio(form, 8000)
        assert actual.dtype == np.float32
        assert actual.shape == (16000,)
        np.testing.assert_allclose(actual, expected)


def test_lazy_decoder(sine):
    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return sine[None, :]

    class Decoder:
        calls = 0

        def get_all_samples(self):
            self.calls += 1
            return SimpleNamespace(data=Tensor(), sample_rate=8000)

    decoder = Decoder()
    assert decode_audio(decoder).shape == (16000,)
    assert decoder.calls == 1


def test_audio_empty_errors_and_window():
    assert decode_audio(np.array([], dtype=np.float32), 8000).size == 0
    for audio, rate in [
        ({}, None),
        (np.zeros(2), None),
        (np.array([np.nan]), 16000),
        (np.array([1], dtype=np.int32), 16000),
    ]:
        with pytest.raises(ValueError):
            decode_audio(audio, rate)
    window, mask = right_aligned_window(np.ones(3, dtype=np.float32), 5)
    assert window.tolist() == [0, 0, 1, 1, 1]
    assert mask.tolist() == [False, False, True, True, True]
    assert not right_aligned_window(np.array([], dtype=np.float32), 5)[1].any()


@pytest.mark.parametrize(
    "text,task",
    [
        ("Is someone speaking?", "voice_activity"),
        ("Has the speaker finished?", "turn_end"),
        ('Did someone say "hello"?', "term"),
        ("What language is spoken?", "language"),
        ("Is this the same speaker?", "same_speaker"),
        ("How many speakers?", "speaker_count"),
    ],
)
def test_routing(text, task):
    assert Question.noul(text).task == task
    assert Question.noul('Did someone say "Hello"?').term == "Hello"


def test_question_errors():
    with pytest.raises(UnsupportedQuestion):
        Question.score("What is happening?")
    with pytest.raises(ValueError):
        Question.choice("What language?", [])
    with pytest.raises(ValueError):
        Question.noul("", task="voice_activity")
    assert Question.score("Custom", task="turn_end").task == "turn_end"


def test_decision_json_and_model(sine):
    model = FakeMarimba()
    questions = [
        Question.noul("Is someone speaking?"),
        Question.score("Has the speaker finished?"),
        Question.choice("What language?", ["English", "French"]),
    ]
    decisions = model(sine, questions, 8000)
    for decision in decisions:
        assert json.loads(json.dumps(decision.to_dict())) == decision.to_dict()
    assert len(model.batch([{"array": sine, "sampling_rate": 8000}], questions)) == 1
    assert model.batch([], questions) == []
    with pytest.raises(ValueError):
        model.batch([], questions, batch_size=0)
    assert model(np.zeros(1, dtype=np.float32), [], 16000) == []


@pytest.mark.parametrize("chunk_size", [1, 777, 4000, 20000])
def test_stream_invariance(chunk_size):
    audio = np.random.default_rng(3).normal(0, 0.1, 17321).astype(np.float32)
    model = FakeMarimba()
    questions = [Question.noul("Is someone speaking?")]
    stream = model.stream(questions, window_seconds=0.5)
    updates = []
    for offset in range(0, len(audio), chunk_size):
        updates.extend(stream.push(audio[offset : offset + chunk_size]))
    updates.extend(stream.flush())
    assert len(updates) == 5
    assert stream.flush() == []
    for update in updates:
        end = round(update.window_end_s * 16000)
        start = max(0, end - 8000)
        expected = model(audio[start:end], questions, 16000)[0]
        assert update.decisions[0].probability == expected.probability
        assert json.loads(json.dumps(update.to_dict())) == update.to_dict()
    assert stream.buffer.size == 128000


def test_stream_int16_and_errors():
    model = FakeMarimba()
    question = [Question.noul("Is someone speaking?")]
    assert model.stream(question).push(np.zeros(4000, dtype=np.int16))[0].decisions
    assert model.stream(question).push(np.array([], dtype=np.float32)) == []
    for kwargs in [{"sample_rate": 8000}, {"hop_seconds": 0}, {"window_seconds": -1}]:
        with pytest.raises(ValueError):
            model.stream(question, **kwargs)


def test_map_fn():
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict(
        {
            "audio": [
                {"array": [0.0] * 32, "sampling_rate": 16000},
                {"array": [0.1] * 32, "sampling_rate": 16000},
            ]
        }
    )
    fn = FakeMarimba().map_fn([Question.noul("Is someone speaking?")])
    mapped = dataset.map(fn, batched=True)
    assert len(mapped["marimba_0"]) == 2
    assert fn({"audio": []}) == {"marimba_0": []}
    with pytest.raises(KeyError):
        fn({})


def test_cli(capsys):
    assert main(["bench", "--fake", "--questions", "3", "--hops", "4", "--warmup", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["hops"] == 4
    assert 0 <= report["p50_ms"] <= report["p95_ms"]
    assert main(["doctor"]) == 0
    assert "selection" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        main(["bench", "--fake", "--hops", "0"])


def test_runtime_order(monkeypatch):
    info = {
        "extras": {"torch": True},
        "cuda": True,
        "cuda_bf16_supported": False,
        "cuda_fp16_supported": True,
        "mps": True,
        "providers": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
        "cuda_free_memory": 1024**3,
        "free_memory": 16 * 1024**3,
    }
    monkeypatch.setattr("marimba.runtime.detect_runtime", lambda: info)
    assert select_backend().device == "cuda"
    assert select_backend().dtype == "float16"
    info["cuda"] = False
    assert select_backend().device == "mps"
    info["mps"] = False
    assert select_backend().backend == "onnx"
    assert select_backend(device="cpu").providers == ("CPUExecutionProvider",)
    info["providers"] = ()
    assert select_backend().device == "cpu"
    with pytest.raises(RuntimeError):
        select_backend(device="cuda")
    with pytest.raises(ValueError):
        select_backend(backend="unknown")
