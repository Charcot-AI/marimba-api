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
