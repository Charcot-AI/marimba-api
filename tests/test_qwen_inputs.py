import importlib
import json
from contextvars import ContextVar
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("installed", [False, True])
def test_layer12_hook_requires_installed_readout(monkeypatch, tmp_path, installed):
    pytest.importorskip("torch")
    module = importlib.import_module("marimba._qwen")
    registered = []

    class Layer:
        def __init__(self, number):
            self.number = number

        def register_forward_hook(self, hook):
            registered.append(self.number)

    thinker = SimpleNamespace(audio_tower=SimpleNamespace(layers=[Layer(i) for i in range(1, 19)]))
    model = SimpleNamespace(thinker=thinker)
    model.eval = lambda: model
    model.requires_grad_ = lambda value: model
    asr = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=SimpleNamespace()))
    monkeypatch.setattr(module, "FrozenASR", lambda *args: asr)
    monkeypatch.setattr(module, "snapshot_download", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "load_file", lambda path: {})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "layer": 18,
                "pooling": "tail",
                "tail_seconds": 2.0,
                "window_seconds": 8.0,
                "sampling_rate": 16000,
                "calibration": "platt",
            }
        )
    )
    paths = {"turn_end/config.json": config, "turn_end/readout.safetensors": tmp_path / "weights"}
    if installed:
        paths["diarization/same_speaker.safetensors"] = tmp_path / "speaker"
    module.QwenEngine(SimpleNamespace(device="cpu", dtype="float32"), paths)
    assert registered == ([18, 12] if installed else [18])


def test_diarization_capture_precision_and_context(engine):
    model, _ = engine
    torch = pytest.importorskip("torch")
    model._captures = ContextVar("test_captures", default=None)
    frames = torch.tensor([[[0.1234567, 1.234567]]])
    model._capture(12, None, None, (frames,))
    captures = {}
    token = model._captures.set(captures)
    try:
        model._capture(12, None, None, (frames,))
    finally:
        model._captures.reset(token)
    np.testing.assert_array_equal(captures[12], frames[0].half().float().numpy())
    assert captures[12].dtype == np.float32
    assert model._captures.get() is None


@pytest.fixture
def engine():
    torch = pytest.importorskip("torch")
    cls = importlib.import_module("marimba._qwen").QwenEngine
    model = cls.__new__(cls)
    model.asr = SimpleNamespace(_build_text_prompt=lambda **kwargs: "prompt")

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    class Thinker:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 128)))

    model.processor = SimpleNamespace(tokenizer=Tokenizer())
    model.thinker = Thinker()
    model._inputs = lambda text, encoded: {
        "input_ids": torch.tensor([[ord(c) for c in text]]),
        "inputs_embeds": torch.zeros((1, len(text), 2)),
        "attention_mask": torch.ones((1, len(text))),
    }
    prefix = {
        "length": 1,
        "ids": torch.tensor([[ord("p")]]),
        "cache": None,
        "rope_deltas": torch.zeros((1, 1)),
    }
    return model, prefix


@pytest.mark.parametrize("cached", [False, True])
def test_language_prefix_cache_is_distinct_from_token_prefix(engine, cached):
    model, prefix = engine
    result = model.language_scores(np.zeros((100, 2)), ["en", "pt"], prefix if cached else None)
    assert result["en"] == pytest.approx(-len(" English") * np.log(128))
    assert result["pt"] == pytest.approx(-len(" Portuguese") * np.log(128))


def test_language_scores_moves_mps_logits_to_cpu(engine):
    model, _ = engine
    torch = pytest.importorskip("torch")
    moved_to_cpu = []

    class MPSLogits:
        device = SimpleNamespace(type="mps")

        def __getitem__(self, key):
            return self

        def cpu(self):
            moved_to_cpu.append(True)
            return torch.zeros((len(" English"), 128))

    class Thinker:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=MPSLogits())

    model.thinker = Thinker()

    result = model.language_scores(np.zeros((100, 2)), ["en"])

    assert moved_to_cpu == [True]
    assert result["en"] == pytest.approx(-len(" English") * np.log(128))


def test_language_empty_and_invalid(engine):
    model, _ = engine
    assert model.language_scores(np.zeros((100, 2)), []) == {}
    with pytest.raises(ValueError, match="Unsupported language"):
        model.language_scores(np.zeros((100, 2)), ["xx"])


def test_wrong_shared_prefix_rejected(engine):
    model, prefix = engine
    prefix["ids"] = prefix["ids"] * 0
    with pytest.raises(ValueError, match="share the encoded audio prefix"):
        model.language_scores(np.zeros((100, 2)), ["en"], prefix)


def test_greedy_transcript_and_five_hypotheses(engine):
    torch = pytest.importorskip("torch")
    model, _ = engine
    beam_counts = []

    class Generator:
        def generate(self, **kwargs):
            beams = kwargs["num_beams"]
            beam_counts.append(beams)
            length = kwargs["input_ids"].shape[1] + 1
            return SimpleNamespace(sequences=torch.zeros((beams, length), dtype=torch.long))

    model.asr.model = Generator()
    model.processor.batch_decode = lambda sequences, **kwargs: [
        f"language English<asr_text>hypothesis {index}" for index in range(len(sequences))
    ]
    encoded = np.zeros((100, 2))
    assert model.transcript(encoded) == "hypothesis 0"
    assert beam_counts == [1]
    beam_counts.clear()
    hypotheses = model.hypotheses(encoded)
    assert len(hypotheses) == 5
    assert hypotheses == [
        "hypothesis 0",
        "hypothesis 0",
        "hypothesis 1",
        "hypothesis 2",
        "hypothesis 3",
    ]
    assert beam_counts == [1, 4]
