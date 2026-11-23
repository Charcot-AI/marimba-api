from collections import deque
from contextvars import ContextVar
from functools import partial
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from marimba import PAIR_LAYOUTS, Question, assemble_pair
from marimba.testing import FakeMarimba

QUESTIONS = [
    Question.score("Has the speaker finished?"),
    Question.noul("Is someone speaking?", seconds=1.0, threshold=0.1),
    Question.score("Did the speaker change?"),
    Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
]


def tone(seconds: float, frequency: float = 220.0, level: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * 16000)) / 16000
    return (level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_pair_layouts_trim_and_repeat() -> None:
    assert dict(PAIR_LAYOUTS) == {"same_speaker": (4.0, 4.0), "speaker_change": (6.0, 2.0)}
    ramp = np.arange(160000, dtype=np.float32) / 160000
    window = assemble_pair("same_speaker", ramp, ramp)
    np.testing.assert_array_equal(window[:64000], ramp[-64000:])
    np.testing.assert_array_equal(window[64000:], ramp[:64000])
    short = assemble_pair("speaker_change", ramp[:16000], ramp[:8000])
    assert short.shape == (128000,)
    np.testing.assert_array_equal(short[:96000], np.tile(ramp[:16000], 6))
    np.testing.assert_array_equal(short[96000:], np.tile(ramp[:8000], 4))
    with pytest.raises(ValueError):
        assemble_pair("turn_end", ramp, ramp)
    with pytest.raises(ValueError):
        assemble_pair("same_speaker", ramp[:0], ramp)


def test_window_decisions_cover_every_window_question() -> None:
    model = FakeMarimba()
    windows = [tone(8.0), np.concatenate((tone(6.0, 110.0), tone(2.0, 330.0)))]
    rows = model.decide_windows(windows, QUESTIONS)
    assert [[d.task for d in row] for row in rows] == [
        ["turn_end", "voice_activity", "speaker_change", "speaker_count"]
    ] * 2
    assert all((d.window_end_s, d.status) == (8.0, "experimental") for row in rows for d in row)
    assert rows[0][1].window_start_s == 7.0 and rows[0][0].window_start_s == 0.0
    assert rows[0][2].probability != rows[1][2].probability
    batched = model.decide_windows(windows, QUESTIONS, batched=True, batch_size=1)
    assert [[d.value for d in row] for row in batched] == [[d.value for d in row] for row in rows]
    assert model.decide_windows([], QUESTIONS) == []
    assert model.decide_windows(windows, []) == [[], []]


def test_window_decisions_reject_bad_input() -> None:
    model = FakeMarimba()
    with pytest.raises(ValueError, match="128000 samples"):
        model.decide_windows([tone(7.9)], QUESTIONS)
    with pytest.raises(ValueError, match="encoder questions only"):
        model.decide_windows([tone(8.0)], [Question.choice("What language?", ["en", "pt"])])
    with pytest.raises(ValueError, match="batch_size"):
        model.decide_windows([tone(8.0)], QUESTIONS, batch_size=0)
    with pytest.raises(ValueError, match="Pair questions"):
        model.decide_pairs(QUESTIONS[0], [(tone(4.0), tone(4.0))])


@pytest.mark.parametrize("task", ["same_speaker", "speaker_change"])
def test_pair_decisions_match_assembled_windows(task: str) -> None:
    model = FakeMarimba()
    question = Question.score("pair", task=task)
    pairs = [(tone(9.0), tone(1.0, level=0.05)), (tone(2.0, 110.0), tone(5.0, 330.0))]
    decisions = model.decide_pairs(question, pairs, batched=True)
    windows = [assemble_pair(task, reference, segment) for reference, segment in pairs]
    expected = [row[0] for row in model.decide_windows(windows, [question])]
    assert [d.probability for d in decisions] == [d.probability for d in expected]
    assert decisions[0].probability != decisions[1].probability


def test_batch_keeps_sequential_default_and_matches_when_batched() -> None:
    model = FakeMarimba()
    audios = [
        {"array": audio, "sampling_rate": 16000}
        for audio in (tone(8.0), tone(3.0), tone(12.0, level=0.05))
    ]
    calls = []
    encode_many = model.encode_many
    model.encode_many = lambda group: calls.append(len(group)) or encode_many(group)  # type: ignore[method-assign]
    sequential = model.batch(audios, QUESTIONS)
    assert calls == []
    batched = model.batch(audios, QUESTIONS, batch_size=2, batched=True)
    assert calls == [2, 1]
    for left, right in zip(sequential, batched, strict=True):
        assert [(d.value, d.window_start_s, d.window_end_s) for d in left] == [
            (d.value, d.window_start_s, d.window_end_s) for d in right
        ]
    # Decoder questions keep the sequential path even when batching is requested.
    model.batch(audios, [Question.choice("What language?", ["en", "pt"])], batched=True)
    assert calls == [2, 1]
    # Any sequence of clips is accepted, including one that cannot be sliced.
    assert len(model.batch(deque(audios), QUESTIONS, batch_size=2, batched=True)) == 3


@pytest.fixture
def engine():
    torch = pytest.importorskip("torch")
    from transformers import WhisperFeatureExtractor

    from marimba._qwen import QwenEngine
    from marimba._vendor.qwen_asr.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from marimba._vendor.qwen_asr.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
        Qwen3ASRThinkerForConditionalGeneration,
    )

    torch.manual_seed(0)
    config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=18,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        output_dim=24,
        n_window=50,
        n_window_infer=800,
        downsample_hidden_size=16,
    )
    config._attn_implementation = "sdpa"
    tower = Qwen3ASRAudioEncoder(config).eval().requires_grad_(False)
    thinker = SimpleNamespace(audio_tower=tower)
    thinker.get_audio_features = MethodType(
        Qwen3ASRThinkerForConditionalGeneration.get_audio_features, thinker
    )
    model = QwenEngine.__new__(QwenEngine)
    model.thinker = thinker
    model.processor = SimpleNamespace(feature_extractor=WhisperFeatureExtractor(feature_size=128))
    model.asr = SimpleNamespace(model=SimpleNamespace(device="cpu", dtype=torch.float32))
    model._captures = ContextVar("test_batch_captures", default=None)
    model._layers = (9, 18, 12)
    for layer in model._layers:
        tower.layers[layer - 1].register_forward_hook(partial(model._capture, layer))
    return model


def test_batched_forward_matches_one_window_per_pass(engine) -> None:
    rng = np.random.default_rng(0)
    windows = [(0.1 * rng.standard_normal(128000)).astype(np.float32) for _ in range(3)]
    batched = engine.encode_batch(windows)
    assert len(batched) == 3
    for window, row in zip(windows, batched, strict=True):
        single = engine.encode(window)
        assert row["features"].keys() == single["features"].keys() == {9, 12, 18}
        for layer, values in single["features"].items():
            assert row["features"][layer].shape == values.shape
            np.testing.assert_allclose(row["features"][layer], values, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(row["audio_features"], single["audio_features"], atol=1e-4)
    # Rows are independent: reordering the batch reorders the output.
    swapped = engine.encode_batch(windows[::-1])
    np.testing.assert_allclose(
        swapped[0]["audio_features"], batched[2]["audio_features"], atol=1e-5
    )
    assert len(engine.encode_batch(windows[:1])) == 1
