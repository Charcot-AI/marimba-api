import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from marimba import Marimba, Question, RuntimeInfo, UnsupportedQuestion
from marimba.pretrained import PretrainedMarimba
from marimba.weights import DEFAULT_MODEL_REVISION, ENCODER_REPO, ENCODER_REVISION

AUDIO_CONFIG = {
    "num_mel_bins": 128,
    "encoder_layers": 18,
    "encoder_attention_heads": 4,
    "encoder_ffn_dim": 64,
    "d_model": 32,
    "output_dim": 24,
    "n_window": 50,
    "n_window_infer": 800,
    "downsample_hidden_size": 16,
}


class EncoderStub:
    def encode(self, window):
        return {"features": {18: np.zeros((100, 2), dtype=np.float32)}, "audio_features": None}

    def turn_end(self, features, mask):
        return 0.6, 0.7


def pretrained(encoder_only: bool) -> PretrainedMarimba:
    manifest = {
        "resolved_revision": DEFAULT_MODEL_REVISION,
        "lanes": {
            "turn_end": {"status": "validated"},
            "language": {"status": "validated"},
            "term": {"status": "validated", "calibration": "none"},
        },
    }
    runtime = RuntimeInfo("fake", "cpu", "float32")
    return PretrainedMarimba(
        runtime, EncoderStub(), manifest, lambda text: True, encoder_only=encoder_only
    )


def test_encoder_only_answers_encoder_questions_and_rejects_decoder_questions() -> None:
    model = pretrained(encoder_only=True)
    audio = np.zeros(16000, dtype=np.float32)
    assert model(audio, [Question.score("Has the speaker finished?")], 16000)[0].value == 0.7
    capabilities = model.capabilities
    assert capabilities["turn_end"].status == "validated"
    for task in ("language", "term", "pii"):
        assert capabilities[task].status == "unavailable"
        assert "audio encoder only" in capabilities[task].evaluation
    for question in (
        Question.choice("What language?", ["en", "pt"]),
        Question.noul('Did someone say "hello"?'),
        Question.noul("Does the audio contain PII?", task="pii"),
    ):
        with pytest.raises(UnsupportedQuestion, match="audio encoder only"):
            model(audio, [Question.score("Has the speaker finished?"), question], 16000)
    assert pretrained(encoder_only=False).capabilities["language"].status == "validated"


def test_from_pretrained_passes_the_encoder_options(monkeypatch, tmp_path) -> None:
    seen = []
    monkeypatch.setattr("marimba.pretrained.load_pretrained", lambda *args: seen.append(args[-3:]))
    Marimba.from_pretrained()
    Marimba.from_pretrained(encoder_only=True)
    Marimba.from_pretrained(encoder_dir=tmp_path)
    Marimba.from_pretrained(encoder_only=True, tensorrt_dir=tmp_path)
    assert seen == [
        (False, None, None),
        (True, None, None),
        (False, tmp_path, None),
        (True, None, tmp_path),
    ]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """A small stand-in for the encoder snapshot: an audio tower and decoder tensors."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from transformers import WhisperFeatureExtractor

    from marimba._vendor.qwen_asr.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from marimba._vendor.qwen_asr.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    torch.manual_seed(0)
    tower = Qwen3ASRAudioEncoder(Qwen3ASRAudioEncoderConfig(**AUDIO_CONFIG)).eval()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tensors = {f"thinker.audio_tower.{name}": value for name, value in tower.state_dict().items()}
    tensors["thinker.model.embed_tokens.weight"] = torch.zeros(8, 4)
    tensors["thinker.lm_head.weight"] = torch.zeros(8, 4)
    save_file(tensors, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(
        json.dumps({"thinker_config": {"audio_config": AUDIO_CONFIG, "text_config": {}}})
    )
    WhisperFeatureExtractor(feature_size=128).save_pretrained(snapshot)
    requests = []

    def download(repo, revision, allow_patterns):
        requests.append((repo, revision, tuple(allow_patterns)))
        return str(snapshot)

    monkeypatch.setattr("marimba.weights.snapshot_download", download)
    return SimpleNamespace(tower=tower, snapshot=snapshot, requests=requests)


def test_export_writes_the_audio_tower_and_nothing_else(checkpoint, tmp_path) -> None:
    import torch
    from safetensors.torch import load_file

    from marimba.weights import export_audio_encoder, load_audio_encoder

    directory = export_audio_encoder(tmp_path / "audio-encoder")
    assert checkpoint.requests == [
        (
            ENCODER_REPO,
            ENCODER_REVISION,
            ("model.safetensors", "config.json", "preprocessor_config.json"),
        )
    ]
    assert sorted(path.name for path in directory.iterdir()) == [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ]
    exported = load_file(directory / "model.safetensors")
    reference = checkpoint.tower.state_dict()
    assert exported.keys() == reference.keys()
    assert all(torch.equal(exported[name], reference[name]) for name in reference)
    config, tensors, files = load_audio_encoder(directory)
    assert (config, files, tensors.keys()) == (AUDIO_CONFIG, directory, reference.keys())
    assert len(checkpoint.requests) == 1

    stale = json.loads((directory / "config.json").read_text())
    stale["source"]["revision"] = "0" * 40
    (directory / "config.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="pinned encoder revision"):
        load_audio_encoder(directory)


def test_a_checkpoint_without_an_audio_tower_is_rejected(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from marimba.weights import audio_tower_tensors

    save_file({"thinker.lm_head.weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="no audio-tower tensors"):
        audio_tower_tensors(tmp_path / "model.safetensors")


@pytest.mark.parametrize("exported", [False, True])
def test_encoder_only_engine_matches_the_thinker_path(checkpoint, tmp_path, exported) -> None:
    import torch

    from marimba._qwen import FrozenAudioEncoder
    from marimba._vendor.qwen_asr.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
    )
    from marimba.weights import export_audio_encoder

    directory = export_audio_encoder(tmp_path / "audio-encoder") if exported else None
    encoder = FrozenAudioEncoder(directory, RuntimeInfo("torch", "cpu", "float32"))
    reference = checkpoint.tower.state_dict()
    loaded = encoder.model.state_dict()
    assert loaded.keys() == reference.keys()
    assert all(torch.equal(loaded[name], reference[name]) for name in reference)
    assert not encoder.model.training
    assert not hasattr(encoder.processor, "tokenizer")

    thinker = SimpleNamespace(audio_tower=checkpoint.tower)
    thinker.get_audio_features = MethodType(
        Qwen3ASRThinkerForConditionalGeneration.get_audio_features, thinker
    )
    rng = np.random.default_rng(0)
    windows = [(0.1 * rng.standard_normal(size)).astype(np.float32) for size in (128000, 48000)]
    features = encoder.processor.feature_extractor(
        windows,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=False,
    )
    with torch.inference_mode():
        expected = thinker.get_audio_features(
            features["input_features"], feature_attention_mask=features["attention_mask"]
        )
        actual = encoder.thinker.get_audio_features(
            features["input_features"], feature_attention_mask=features["attention_mask"]
        )
    assert torch.equal(actual, expected)


def test_export_encoder_command(monkeypatch, tmp_path, capsys) -> None:
    from marimba.cli import main

    def export(directory):
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"12345")
        return directory

    monkeypatch.setattr("marimba.weights.export_audio_encoder", export)
    assert main(["export-encoder", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out) == {"directory": str(tmp_path / "out"), "bytes": 5}


def test_serve_passes_the_encoder_options(monkeypatch, tmp_path) -> None:
    from marimba import cli
    from marimba.testing import FakeMarimba

    seen = []

    def load(**kwargs):
        seen.append((kwargs["encoder_only"], kwargs["encoder_dir"]))
        return FakeMarimba()

    monkeypatch.setattr(cli.Marimba, "from_pretrained", load)
    monkeypatch.setattr("marimba.server.serve", lambda model, config, **kwargs: None)
    monkeypatch.delenv("MARIMBA_ENCODER_DIR", raising=False)
    cli.main(["serve"])
    cli.main(["serve", "--encoder-only"])
    cli.main(["serve", "--encoder-dir", str(tmp_path)])
    assert seen == [(False, None), (True, None), (False, tmp_path)]


@pytest.mark.real_model
def test_real_model_encoder_only_matches_the_full_model(tmp_path) -> None:
    from marimba.weights import export_audio_encoder

    rng = np.random.default_rng(0)
    windows = [(0.05 * rng.standard_normal(128000)).astype(np.float32) for _ in range(2)]
    questions = [
        Question.score("Has the speaker finished?"),
        Question.noul("Is someone speaking?"),
        Question.noul("Is someone speaking?", seconds=1.0),
        Question.score("Did the speaker change?"),
        Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
    ]

    def answers(**options):
        model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32", **options)
        return [
            [(d.probability, d.probabilities) for d in row]
            for batched in (True, False)
            for row in model.decide_windows(windows, questions, batched=batched)
        ]

    full = answers()
    assert answers(encoder_only=True) == full
    assert answers(encoder_dir=export_audio_encoder(tmp_path / "audio-encoder")) == full
