"""Full public-API, single-question L4 latency with a frozen matrix corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from marimba import Marimba, Question

PATHS = ("turn_end", "voice_activity", "language", "term")
COUNTS = dict(zip(PATHS, (300, 300, 300, 200), strict=True))
WARMUP = 20
MODEL_REVISION = "bc980b4e6eb34e7fea96ea8b0884c2093a845dc5"
ENCODER_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DATASET_REVISIONS = {
    "pipecat-ai/smart-turn-data-v3.2-train": "f8c95d284595defb6b88a715c356cbc5c31b7ad1",
    "google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
}
Audio = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: Sequence[float]) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if (
        samples.ndim != 1
        or not samples.size
        or not np.isfinite(samples).all()
        or (samples < 0).any()
    ):
        raise ValueError("Latency samples must be a nonempty, finite, nonnegative vector")
    rng = np.random.default_rng(42)
    resampled = rng.choice(samples, size=(1000, len(samples)), replace=True)
    bootstrap_p95 = np.percentile(resampled, 95, axis=1, method="linear")
    return {
        "n": len(samples),
        "p50_ms": float(np.percentile(samples, 50, method="linear")),
        "p95_ms": float(np.percentile(samples, 95, method="linear")),
        "p95_ci95_ms": np.percentile(bootstrap_p95, [2.5, 97.5], method="linear").tolist(),
        "calls_ms": samples.tolist(),
    }


def questions() -> dict[str, Question]:
    return {
        "turn_end": Question.score("Has the speaker finished?", task="turn_end"),
        "voice_activity": Question.noul("Is someone speaking?", task="voice_activity"),
        "language": Question.choice(
            "What language is being spoken?", ["en", "pt", "es", "fr", "de"], task="language"
        ),
        "term": Question.noul('Did someone say "marimba"?', task="term"),
    }


def path_indices(records: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {name: [] for name in PATHS}
    for index, row in enumerate(records):
        source = row["source"]
        if source == "synthetic":
            if row["lane"] != "vad_negative" or row.get("seed") != 4412:
                raise ValueError("Unexpected synthetic corpus source")
        elif (
            source not in DATASET_REVISIONS
            or row.get("revision") != DATASET_REVISIONS[source]
            or row.get("split") != ("train" if source.startswith("pipecat-ai/") else "validation")
        ):
            raise ValueError("Expected pinned, non-test public matrix sources")
        if row["lane"] == "turn_end":
            result["turn_end"].append(index)
        if row["lane"] == "vad_negative" or row.get("vad") == 1:
            result["voice_activity"].append(index)
        if row["lane"] == "decoder":
            result["language"].append(index)
            result["term"].append(index)
    if any(len(indices) != 100 for indices in result.values()):
        raise ValueError("Expected exactly 100 matrix clips per path")
    return result


def load_inputs(
    corpus: Path, records_path: Path, expected_sha256: str
) -> tuple[dict[str, list[Audio]], dict[str, Any]]:
    if sha256(corpus) != expected_sha256:
        raise ValueError("Matrix corpus SHA256 mismatch")
    metadata = json.loads(records_path.read_text())
    if metadata["rate"] != 16000 or metadata["window_seconds"] != 8:
        raise ValueError("Expected eight-second, 16 kHz matrix windows")
    indices = path_indices(metadata["records"])
    with np.load(corpus, allow_pickle=False) as archive:
        samples, masks = archive["audio"], archive["mask"]
    if (
        samples.shape != (len(metadata["records"]), 128000)
        or samples.dtype != np.float32
        or masks.shape != samples.shape
        or masks.dtype != np.bool_
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Invalid matrix audio or mask arrays")
    audio = {}
    clips = {}
    for name, selected in indices.items():
        audio[name] = [{"array": samples[index], "sampling_rate": 16000} for index in selected]
        clips[name] = []
        for index in selected:
            row = metadata["records"][index]
            clip = {
                key: row[key]
                for key in ("source", "revision", "split", "file", "id", "seed")
                if key in row
            }
            clip.update(
                corpus_index=index,
                window_sha256=hashlib.sha256(samples[index].tobytes()).hexdigest(),
                original_valid_samples=int(masks[index].sum()),
            )
            clips[name].append(clip)
    return audio, {
        "format_version": 1,
        "dataset_revisions": DATASET_REVISIONS,
        "corpus_sha256": expected_sha256,
        "audio_input": "Preloaded public mappings: float32 array, sampling_rate=16000.",
        "padding": "Existing right-aligned eight-second matrix windows, including left padding.",
        "mask_policy": (
            "Original masks are provenance only; the public API recomputes its own masks."
        ),
        "cycle": "Warmup starts at clip 0; measured calls continue at clip 20 modulo 100.",
        "paths": clips,
    }


def measure_call(
    model: Callable[..., Any],
    audio: Audio,
    question: Question,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    synchronize()
    started = clock()
    decisions = model(audio, [question])
    synchronize()
    elapsed = (clock() - started) * 1000
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Expected exactly one returned Decision per call")
    if decisions[0].status not in ("validated", "experimental"):
        raise ValueError("The requested inference path is unavailable")
    return elapsed


def run_protocol(
    model: Callable[..., Any],
    audio: Mapping[str, Sequence[Audio]],
    synchronize: Callable[[], None],
    *,
    warmup: int = WARMUP,
    counts: Mapping[str, int] = COUNTS,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, list[float]]:
    if set(counts) != set(PATHS) or any(counts[name] <= 0 for name in PATHS) or warmup < 0:
        raise ValueError("Invalid per-path counts or warmup")
    if any(not audio.get(name) for name in PATHS):
        raise ValueError("Every path requires a nonempty clip list")
    prompts = questions()
    result: dict[str, list[float]] = {name: [] for name in PATHS}
    for round_index in range(warmup):
        for name in PATHS:
            measure_call(
                model,
                audio[name][round_index % len(audio[name])],
                prompts[name],
                synchronize,
                clock,
            )
        if progress:
            progress("warmup", round_index + 1)
    for round_index in range(max(counts.values())):
        for name in PATHS:
            if round_index >= counts[name]:
                continue
            clip = audio[name][(warmup + round_index) % len(audio[name])]
            result[name].append(measure_call(model, clip, prompts[name], synchronize, clock))
        if progress:
            progress("measured", round_index + 1)
    return result


def main() -> None:
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-ids", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        parser.error("code-commit must be a full immutable commit SHA")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA GPU")
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA L4":
        raise RuntimeError("Expected NVIDIA L4")
    torch.set_num_threads(4)
    audio, clip_ids = load_inputs(args.corpus, args.records, args.corpus_sha256)
    args.clip_ids.write_text(json.dumps(clip_ids, indent=2, allow_nan=False) + "\n")
    model = Marimba.from_pretrained(
        backend="torch", device="cuda", dtype="auto", revision=MODEL_REVISION
    )
    if model.runtime.dtype != "bfloat16":
        raise RuntimeError("CUDA auto must resolve to bfloat16")
    report = {
        "format_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": args.code_commit,
        "package_archive_sha256": args.package_sha256,
        "runner_sha256": sha256(Path(__file__)),
        "model_revision": MODEL_REVISION,
        "encoder": {"repo": "Qwen/Qwen3-ASR-0.6B", "revision": ENCODER_REVISION},
        "runtime": model.runtime.to_dict(),
        "hardware": {
            "gpu": gpu,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "gpu_count": torch.cuda.device_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "versions": {
            "python": platform.python_version(),
            "marimba": importlib.metadata.version("marimba-audio"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "qwen_asr": {"implementation": "vendored", "upstream_version": "0.0.6"},
        },
        "clip_ids_file": args.clip_ids.name,
        "clip_ids_sha256": sha256(args.clip_ids),
        "protocol": {
            "batch_size": 1,
            "window_seconds": 8.0,
            "sampling_rate": 16000,
            "warmup_calls_per_path": WARMUP,
            "measured_calls_per_path": COUNTS,
            "order": list(PATHS),
            "interleaving": "Round-robin; omit term after its 200th measured call.",
            "timer": (
                "perf_counter around model(audio, [question]); "
                "CUDA synchronize before start and before stop."
            ),
            "input": (
                "Preloaded audio mappings; no file I/O inside the timer; "
                "includes public input normalization."
            ),
            "bootstrap": {
                "resamples": 1000,
                "seed": 42,
                "generator": "numpy.random.default_rng, reset per path",
                "interval": (
                    "Percentile bootstrap over calls: 2.5th/97.5th percentiles of resampled p95."
                ),
                "percentile_method": "linear",
            },
        },
        "questions": {
            name: {"text": q.text, "kind": q.kind, "task": q.task, "options": q.options}
            for name, q in questions().items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def progress(phase: str, count: int) -> None:
        if count % 10 == 0:
            print(phase, count, flush=True)

    values = run_protocol(model, audio, torch.cuda.synchronize, progress=progress)
    report.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        paths={name: summarize(values[name]) for name in PATHS},
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("L4_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
