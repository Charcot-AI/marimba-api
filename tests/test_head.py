import json

import pytest


def test_grapheme_positive_option():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from marimba._head import GraphemeTokenizer

    ids, mask = GraphemeTokenizer()("sim")
    assert ids.tolist() == [[2, 22, 12, 16, 3]]
    assert mask.dtype == torch.bool and mask.all()
    long_ids, _ = GraphemeTokenizer()("a" * 100)
    assert long_ids.shape == (1, 64) and long_ids[0, -1] == 3


def test_head_masks_padded_audio():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from marimba._head import MarimbaHead

    torch.manual_seed(0)
    head = MarimbaHead().eval()
    audio = torch.randn(1, 10, 896)
    mask = torch.tensor([[False] * 5 + [True] * 5])
    question = torch.randn(1, 4, 256)
    option = torch.randn(1, 5, 256)
    changed = audio.clone()
    changed[:, :5] = 1000
    with torch.inference_mode():
        first = head(
            audio, mask, question, torch.ones(1, 4).bool(), option, torch.ones(1, 5).bool()
        )
        second = head(
            changed, mask, question, torch.ones(1, 4).bool(), option, torch.ones(1, 5).bool()
        )
    assert first.shape == (1,)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_invalid_head_configuration(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from marimba._head import SpeechReadout

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"layer": 18}))
    with pytest.raises(ValueError, match="configuration"):
        SpeechReadout({"speech_head/config.json": path}, "cpu")


def test_cached_option_is_reusable_with_bit_identical_cpu_scores(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np

    from marimba import _head

    torch.manual_seed(0)
    model = _head.SpeechModel().eval().requires_grad_(False)
    with torch.no_grad():
        model.text.positions.zero_()
        model.text_options.positions.zero_()
    monkeypatch.setattr(_head, "SpeechModel", lambda: model)
    monkeypatch.setattr(_head, "load_file", lambda _: model.state_dict())
    monkeypatch.setattr(_head.WhisperTokenizer, "from_pretrained", lambda *a, **kw: None)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "layer": 9,
                "frames": 100,
                "dimension_audio": 896,
                "dimension": 256,
                "heads": 4,
                "text_layers": 2,
                "tail_pool_seconds": 0,
                "positive_option": "yes",
                "tokenizer_repo": _head.TEXT_REPO,
                "tokenizer_revision": _head.TEXT_REVISION,
            }
        )
    )
    readout = _head.SpeechReadout(
        {"speech_head/config.json": config, "speech_head/model.safetensors": tmp_path}, "cpu"
    )
    cached = readout.option
    assert not cached.is_inference() and not cached.requires_grad
    assert cached._version >= 0
    ids, mask = _head.GraphemeTokenizer()("sim")
    with torch.inference_mode():
        original = model.text_options(ids, mask)
    assert torch.equal(cached, original)
    question = torch.randn(1, 4, 256)
    readout.question_features = lambda _: (question, torch.ones(1, 4, dtype=torch.bool))
    audio = torch.randn(100, 896)
    audio_mask = np.ones(16000, dtype=np.bool_)
    expected = readout.score(audio, audio_mask, "Is someone speaking?")
    readout.option = original
    actual = readout.score(audio, audio_mask, "Is someone speaking?")
    assert actual == expected
    assert np.isfinite(actual)
