"""Public-API CPU or L4 latency with a frozen matrix corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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
ALL_PATHS = (*PATHS, "turn_vad")
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


def path_indices(
    records: Sequence[Mapping[str, Any]], paths: Sequence[str] = PATHS
) -> dict[str, list[int]]:
    if not paths or len(set(paths)) != len(paths) or set(paths) - set(ALL_PATHS):
        raise ValueError("Invalid inference paths")
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
    result["turn_vad"] = result["turn_end"]
    return {name: result[name] for name in paths}


def load_inputs(
    corpus: Path, records_path: Path, expected_sha256: str, paths: Sequence[str] = PATHS
) -> tuple[dict[str, list[Audio]], dict[str, Any]]:
    if sha256(corpus) != expected_sha256:
        raise ValueError("Matrix corpus SHA256 mismatch")
    metadata = json.loads(records_path.read_text())
    if metadata["rate"] != 16000 or metadata["window_seconds"] != 8:
        raise ValueError("Expected eight-second, 16 kHz matrix windows")
    indices = path_indices(metadata["records"], paths)
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
    question: Question | Sequence[Question],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    synchronize()
    started = clock()
    prompts = [question] if isinstance(question, Question) else list(question)
    decisions = model(audio, prompts)
    synchronize()
    elapsed = (clock() - started) * 1000
    if not prompts or not isinstance(decisions, list) or len(decisions) != len(prompts):
        raise ValueError("Expected one returned Decision per question")
    if any(decision.status not in ("validated", "experimental") for decision in decisions):
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
    paths = tuple(counts)
    if (
        not paths
        or set(paths) - set(ALL_PATHS)
        or any(n <= 0 for n in counts.values())
        or warmup < 0
    ):
        raise ValueError("Invalid per-path counts or warmup")
    if any(not audio.get(name) for name in paths):
        raise ValueError("Every path requires a nonempty clip list")
    prompts = questions()
    selected = {
        name: [prompts["turn_end"], prompts["voice_activity"]]
        if name == "turn_vad"
        else [prompts[name]]
        for name in paths
    }
    result: dict[str, list[float]] = {name: [] for name in paths}
    for round_index in range(warmup):
        for name in paths:
            measure_call(
                model,
                audio[name][round_index % len(audio[name])],
                selected[name],
                synchronize,
                clock,
            )
        if progress:
            progress("warmup", round_index + 1)
    for round_index in range(max(counts.values())):
        for name in paths:
            if round_index >= counts[name]:
                continue
            clip = audio[name][(warmup + round_index) % len(audio[name])]
            result[name].append(measure_call(model, clip, selected[name], synchronize, clock))
        if progress:
            progress("measured", round_index + 1)
    return result


def cpu_metadata() -> dict[str, Any]:
    info = Path("/proc/cpuinfo")
    model = platform.processor()
    if info.exists():
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in info.read_text().splitlines()
                if line.startswith("model name")
            ),
            model,
        )
    return {"cpu_model": model, "vcpus": os.cpu_count(), "load_average": os.getloadavg()}


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
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--paths", nargs="+", choices=ALL_PATHS, default=list(PATHS))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        parser.error("code-commit must be a full immutable commit SHA")
    if args.threads <= 0 or len(set(args.paths)) != len(args.paths):
        parser.error("threads must be positive and paths must be distinct")
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one CUDA GPU")
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else None
    if args.device == "cuda" and gpu != "NVIDIA L4":
        raise RuntimeError("Expected NVIDIA L4")
    torch.set_num_threads(args.threads)
    counts = {name: COUNTS.get(name, 300) for name in args.paths}
    audio, clip_ids = load_inputs(args.corpus, args.records, args.corpus_sha256, args.paths)
    args.clip_ids.write_text(json.dumps(clip_ids, indent=2, allow_nan=False) + "\n")
    model = Marimba.from_pretrained(
        backend="torch", device=args.device, dtype="auto", revision=MODEL_REVISION
    )
    expected_dtype = "bfloat16" if args.device == "cuda" else "float32"
    if model.runtime.dtype != expected_dtype:
        raise RuntimeError(f"{args.device} auto must resolve to {expected_dtype}")
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
            ).strip()
            if args.device == "cuda"
            else None,
            "gpu_count": torch.cuda.device_count() if args.device == "cuda" else 0,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            **(cpu_metadata() if args.device == "cpu" else {}),
        },
        "versions": {
            "python": platform.python_version(),
            "marimba": importlib.metadata.version("marimba"),
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
            "measured_calls_per_path": counts,
            "order": args.paths,
            "interleaving": "Round-robin; omit a path when its call count is reached.",
            "timer": (
                "perf_counter around model(audio, questions); "
                + (
                    "CUDA synchronize before start and before stop."
                    if args.device == "cuda"
                    else "Synchronous CPU execution."
                )
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
    if "turn_vad" in args.paths:
        report["questions"]["turn_vad"] = [
            report["questions"]["turn_end"],
            report["questions"]["voice_activity"],
        ]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def progress(phase: str, count: int) -> None:
        if count % 10 == 0:
            print(phase, count, flush=True)

    sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    values = run_protocol(model, audio, sync, counts=counts, progress=progress)
    report.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        paths={name: summarize(values[name]) for name in args.paths},
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("L4_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
