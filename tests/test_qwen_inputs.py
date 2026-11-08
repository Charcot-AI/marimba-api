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


@pytest.mark.parametrize("length", [43, 200, 239])
def test_truncated_tower_matches_full_taps_and_skips_projection(length):
    torch = pytest.importorskip("torch")
    from contextvars import ContextVar
    from functools import partial

    from marimba._qwen import QwenEngine
    from marimba._vendor.qwen_asr.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from marimba._vendor.qwen_asr.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=16,
        d_model=8,
        encoder_layers=18,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        output_dim=12,
        n_window=50,
        n_window_infer=400,
        max_source_positions=64,
        conv_chunksize=1,
    )
    config._attn_implementation = "sdpa"
    tower = Qwen3ASRAudioEncoder(config).eval()
    model = QwenEngine.__new__(QwenEngine)
    model._captures = ContextVar("test_captures", default=None)
    model.thinker = SimpleNamespace(audio_tower=tower)
    handles = [
        tower.layers[layer - 1].register_forward_hook(partial(model._capture, layer))
        for layer in (9, 18)
    ]
    inputs = torch.randn(1, 16, length)
    mask = torch.ones(1, length, dtype=torch.long)
    full = {}
    token = model._captures.set(full)
    with torch.inference_mode():
        tower(inputs[0], feature_lens=mask.sum(1))
    model._captures.reset(token)
    projection_calls = []
    for name in ("ln_post", "proj1", "act", "proj2"):
        handles.append(
            getattr(tower, name).register_forward_hook(lambda *args: projection_calls.append(True))
        )
    for depth in (9, 18):
        captures = {}
        token = model._captures.set(captures)
        with torch.inference_mode():
            model._encode_truncated(inputs, mask, depth)
        model._captures.reset(token)
        assert set(captures) == ({9} if depth == 9 else {9, 18})
        for layer in captures:
            np.testing.assert_array_equal(captures[layer], full[layer])
    assert not projection_calls and model._captures.get() is None
    for handle in handles:
        handle.remove()
