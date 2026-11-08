import json

import numpy as np
import pytest
import soundfile as sf

from marimba.cli import benchmark, demo_questions, main
from marimba.testing import FakeMarimba


def test_benchmark_json_fake(tmp_path, capsys):
    output = tmp_path / "bench.json"
    assert (
        main(
            [
                "bench",
                "--fake",
                "--per-question",
                "--hops",
                "2",
                "--warmup",
                "1",
                "--json",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert len(report["questions"]) == len(report["per_question"]) == 5
    assert [row["task"] for row in report["questions"]] == [
        "turn_end",
        "voice_activity",
        "same_speaker",
        "language",
        "term",
    ]
    assert report["questions"][3]["options"] == ["en", "pt", "es"]
    assert report["encoder_only"] is None
    assert report["full_set"]["window_end_s"] == [8.5, 8.75]
    assert len(report["full_set"]["samples_ms"]) == 2
    assert report["p50_ms"] <= report["p95_ms"]


def test_matching_card_windows_and_encoder(tmp_path, monkeypatch, capsys):
    class Engine:
        def __init__(self):
            self.windows = []

        def encode(self, window):
            self.windows.append(window.copy())

    class RecordingFake(FakeMarimba):
        revision = "test-revision"

        def __init__(self):
            super().__init__()
            self._engine = Engine()

        def decide_window(self, window, *args):
            self._engine.encode(window)
            return super().decide_window(window, *args)

    model = RecordingFake()
    monkeypatch.setattr("marimba.cli.Marimba.from_pretrained", lambda **kwargs: model)
    audio = np.linspace(-0.2, 0.2, 128000, dtype=np.float32)
    path = tmp_path / "speech.wav"
    sf.write(path, audio, 16000, subtype="FLOAT")
    assert (
        main(["bench", "--audio", str(path), "--per-question", "--hops", "2", "--warmup", "1"]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["revision"] == "test-revision"
    assert len(model._engine.windows) == 7 * 3
    first = np.concatenate((audio[4000:], audio[:4000]))
    for offset in range(0, 21, 3):
        np.testing.assert_array_equal(model._engine.windows[offset], first)
    assert report["encoder_only"]["window_end_s"] == [8.5, 8.75]
    assert len(report["encoder_in_full_set"]["samples_ms"]) == 2
    assert 0 <= report["encoder_in_full_set"]["share_p50"] <= 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--threads", "0"],
        ["--json", "missing-parent/output.json"],
        ["--audio", "nonexistent.wav"],
        ["--questions", "0"],
        ["--warmup", "-1"],
    ],
)
def test_bench_invalid_arguments(arguments):
    with pytest.raises(SystemExit):
        main(["bench", "--fake", *arguments])


def test_short_audio_rejected(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(10), 16000)
    with pytest.raises(SystemExit):
        main(["bench", "--fake", "--audio", str(path)])


def test_encode_wrapper_restored_after_error():
    class Engine:
        def encode(self, window):
            raise RuntimeError("synthetic encoder failure")

    class FailingFake(FakeMarimba):
        _engine = Engine()

        def decide_window(self, window, *args):
            self._engine.encode(window)

    model = FailingFake()
    original = model._engine.encode
    with pytest.raises(RuntimeError, match="synthetic encoder"):
        benchmark(model, np.zeros(128000, dtype=np.float32), demo_questions(), 0, 1, True)
    assert model._engine.encode == original
