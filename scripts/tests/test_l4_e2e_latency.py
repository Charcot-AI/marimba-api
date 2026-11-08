import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def bench():
    path = Path(__file__).parents[1] / "l4_e2e_latency.py"
    spec = importlib.util.spec_from_file_location("l4_e2e_latency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_linear_percentile_and_seeded_bootstrap(bench):
    samples = list(range(1, 301))
    result = bench.summarize(samples)
    rng = np.random.default_rng(42)
    expected = np.percentile(
        np.percentile(rng.choice(samples, size=(1000, 300)), 95, axis=1, method="linear"),
        [2.5, 97.5],
        method="linear",
    )
    assert result["n"] == 300
    assert result["p50_ms"] == 150.5
    assert result["p95_ms"] == pytest.approx(285.05)
    assert result["p95_ci95_ms"] == expected.tolist()
    assert bench.summarize([5])["p95_ci95_ms"] == [5, 5]


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [-1], [[1, 2]]])
def test_summary_rejects_invalid_samples(bench, samples):
    with pytest.raises(ValueError, match="Latency samples"):
        bench.summarize(samples)


def test_timer_includes_return_and_cuda_completion(bench):
    events = []
    times = iter((1.0, 1.025))

    def sync():
        events.append("sync")

    def clock():
        events.append("clock")
        return next(times)

    question = bench.questions()["turn_end"]
    audio = {"array": np.zeros(128000, dtype=np.float32), "sampling_rate": 16000}

    def model(received_audio, received_questions):
        assert received_audio is audio
        assert received_questions == [question]
        events.append("model returned list")
        return [SimpleNamespace(status="validated")]

    assert bench.measure_call(model, audio, question, sync, clock) == pytest.approx(25)
    assert events == ["sync", "clock", "model returned list", "sync", "clock"]


@pytest.mark.parametrize("result", [None, [], [1, 2], [SimpleNamespace(status="unavailable")]])
def test_timer_rejects_missing_decisions(bench, result):
    with pytest.raises(ValueError):
        bench.measure_call(lambda *_: result, {}, bench.questions()["term"], lambda: None)


def test_exact_protocol_counts_order_and_clip_cycles(bench):
    calls = []
    audio = {name: [{"clip_index": index} for index in range(100)] for name in bench.PATHS}

    def model(clip, prompts):
        assert len(prompts) == 1
        calls.append((prompts[0].task, clip["clip_index"]))
        return [SimpleNamespace(status="validated")]

    result = bench.run_protocol(model, audio, lambda: None)
    assert len(calls) == 1180
    warmup, measured = calls[:80], calls[80:]
    assert warmup == [(name, index) for index in range(20) for name in bench.PATHS]
    assert measured == [
        (name, (20 + index) % 100)
        for index in range(300)
        for name in bench.PATHS
        if index < bench.COUNTS[name]
    ]
    assert {name: len(values) for name, values in result.items()} == bench.COUNTS
    assert bench.questions()["language"].options == ("en", "pt", "es", "fr", "de")
    assert bench.questions()["term"].term == "marimba"


def test_protocol_rejects_empty_audio_before_inference(bench):
    with pytest.raises(ValueError, match="nonempty clip list"):
        bench.run_protocol(lambda *_: pytest.fail("must not infer"), {}, lambda: None)
    with pytest.raises(ValueError, match="Invalid per-path counts"):
        bench.run_protocol(lambda *_: None, {}, lambda: None, counts={})


def matrix_records(bench):
    smart, fleurs = bench.DATASET_REVISIONS
    return (
        [
            {
                "source": smart,
                "revision": bench.DATASET_REVISIONS[smart],
                "split": "train",
                "lane": "turn_end",
            }
            for _ in range(100)
        ]
        + [
            {
                "source": fleurs,
                "revision": bench.DATASET_REVISIONS[fleurs],
                "split": "validation",
                "lane": "decoder",
                "vad": int(i < 50),
            }
            for i in range(100)
        ]
        + [{"source": "synthetic", "seed": 4412, "lane": "vad_negative"} for _ in range(50)]
    )


def test_matrix_lane_subsets_match_parity_selection(bench):
    selected = bench.path_indices(matrix_records(bench))
    assert selected["turn_end"] == list(range(100))
    assert selected["voice_activity"] == list(range(100, 150)) + list(range(200, 250))
    assert selected["language"] == selected["term"] == list(range(100, 200))


def test_matrix_rejects_empty_or_test_split(bench):
    with pytest.raises(ValueError, match="100 matrix clips"):
        bench.path_indices([])
    records = matrix_records(bench)
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="non-test public"):
        bench.path_indices(records)
    records[0]["split"] = "train"
    records[-1]["seed"] = 999
    with pytest.raises(ValueError, match="synthetic corpus"):
        bench.path_indices(records)


def test_corpus_hash_rejection(bench, tmp_path):
    corpus = tmp_path / "corpus.npz"
    np.savez_compressed(corpus, audio=np.zeros((1, 128000), dtype=np.float32))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bench.load_inputs(corpus, tmp_path / "records.json", "0" * 64)


def test_cpu_combined_path_protocol(bench):
    paths = ("turn_end", "voice_activity", "turn_vad")
    counts = dict.fromkeys(paths, 300)
    audio = {name: [{"clip_index": index} for index in range(100)] for name in paths}
    calls = []

    def model(clip, prompts):
        tasks = tuple(q.task for q in prompts)
        calls.append((tasks, clip["clip_index"]))
        return [SimpleNamespace(status="validated") for q in prompts]

    result = bench.run_protocol(model, audio, lambda: None, counts=counts)
    tasks = [("turn_end",), ("voice_activity",), ("turn_end", "voice_activity")]
    assert calls == [(group, index % 100) for index in range(320) for group in tasks]
    assert {name: len(values) for name, values in result.items()} == counts
    selected = bench.path_indices(matrix_records(bench), paths)
    assert selected["turn_vad"] == selected["turn_end"]
    hardware = bench.cpu_metadata()
    assert hardware["cpu_model"] and hardware["vcpus"] > 0
    assert len(hardware["load_average"]) == 3


def test_invalid_path_selection(bench):
    for paths in [[], ["unknown"], ["turn_end", "turn_end"]]:
        with pytest.raises(ValueError, match="Invalid inference paths"):
            bench.path_indices(matrix_records(bench), paths)
