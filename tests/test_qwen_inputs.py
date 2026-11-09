import importlib
from types import SimpleNamespace

import numpy as np
import pytest


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
