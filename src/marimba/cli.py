"""Runtime diagnostics and synchronized per-hop benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .audio import AudioArray, Mask, decode_audio
from .decision import Decision
from .model import Marimba
from .questions import Question
from .runtime import detect_runtime, select_backend, synchronize
from .stream import Stream
from .testing import FakeMarimba


def demo_questions() -> list[Question]:
    return [
        Question.score("Has the speaker finished their turn?", task="turn_end"),
        Question.noul("Is someone speaking?"),
        Question.choice("What language is being spoken?", ["en", "pt", "es"]),
        Question.noul('Did someone say "marimba"?'),
    ]


def percentiles(samples: list[float]) -> dict[str, Any]:
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def measure(stream: Stream, audio: AudioArray, warmup: int, hops: int) -> dict[str, Any]:
    stream.ingest(audio)
    timings, ends = [], []
    for index in range(warmup + hops):
        positions = np.arange(index * 4000, (index + 1) * 4000) % len(audio)
        update = stream.push(audio[positions])[0]
        if index >= warmup:
            timings.append(update.latency_ms)
            ends.append(update.window_end_s)
    return {**percentiles(timings), "window_end_s": ends, "encoder_included": True}


class EncoderOnly:
    def __init__(self, model: Marimba) -> None:
        self.runtime = model.runtime
        self.engine: Any = getattr(model, "_engine", None)

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        self.engine.encode(window)
        return []


def benchmark(
    model: Marimba,
    audio: AudioArray,
    questions: list[Question],
    warmup: int,
    hops: int,
    per_question: bool,
) -> dict[str, Any]:
    engine: Any = getattr(model, "_engine", None)
    encoder_samples: list[float] = []
    encode: Any = getattr(engine, "encode", None)

    def timed_encode(window: AudioArray) -> Any:
        synchronize(model.runtime)
        started = perf_counter()
        encoded = encode(window)
        synchronize(model.runtime)
        encoder_samples.append((perf_counter() - started) * 1000)
        return encoded

    if per_question and encode is not None:
        engine.encode = timed_encode
    try:
        full = measure(model.stream(questions), audio, warmup, hops)
    finally:
        if per_question and encode is not None:
            engine.encode = encode
    result: dict[str, Any] = {"full_set": full, "per_question": [], "encoder_only": None}
    if per_question:
        for question in questions:
            result["per_question"].append(
                {
                    "question": asdict(question),
                    **measure(model.stream([question]), audio, warmup, hops),
                }
            )
        if encode is not None:
            result["encoder_only"] = measure(Stream(EncoderOnly(model), []), audio, warmup, hops)
            samples = encoder_samples[warmup:]
            if len(samples) != hops:
                raise RuntimeError("Expected one encoder pass per full-set hop")
            shares = np.asarray(samples) / np.asarray(full["samples_ms"])
            result["encoder_in_full_set"] = {
                **percentiles(samples),
                "share_p50": float(np.percentile(shares, 50)),
                "share_p95": float(np.percentile(shares, 95)),
                "share_definition": "paired encoder time / full-set time at each measured hop",
            }
    return result


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marimba")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    bench = commands.add_parser("bench")
    bench.add_argument("--questions", type=int, default=4, help="Count from the four demo cards")
    bench.add_argument("--hops", type=int, default=40)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--backend", default="auto")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--fake", action="store_true")
    bench.add_argument("--dtype", default="auto")
    bench.add_argument("--revision", default=None)
    bench.add_argument("--threads", type=int)
    bench.add_argument("--audio", type=Path, help="Public speech file, at least eight seconds")
    bench.add_argument("--json", type=Path, dest="json_path", help="Write the JSON report")
    bench.add_argument(
        "--per-question",
        action="store_true",
        help="Also measure each card alone and the shared encoder",
    )
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = detect_runtime()
        try:
            report["selection"] = select_backend().to_dict()
        except RuntimeError as error:
            report["selection"] = None
            report["message"] = str(error)
        print(json.dumps(report))
        return 0
    if args.questions <= 0 or args.hops <= 0 or args.warmup < 0:
        parser.error("questions and hops must be positive; warmup must be nonnegative")
    if args.threads is not None and args.threads <= 0:
        parser.error("threads must be positive")
    if args.json_path is not None and not args.json_path.parent.is_dir():
        parser.error("JSON output parent directory must exist")
    if args.audio is None:
        audio: AudioArray = np.zeros(128000, dtype=np.float32)
    else:
        try:
            audio = decode_audio(args.audio)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(f"Could not decode benchmark audio: {error}")
        if len(audio) < 128000:
            parser.error("Benchmark audio must contain at least eight seconds")
        audio = audio[-128000:]
    if args.threads is not None:
        if args.fake:
            parser.error("--threads requires a real torch benchmark")
        importlib.import_module("torch").set_num_threads(args.threads)
    model = (
        FakeMarimba()
        if args.fake
        else Marimba.from_pretrained(
            backend=args.backend, device=args.device, dtype=args.dtype, revision=args.revision
        )
    )
    templates = demo_questions()
    questions = [templates[index % len(templates)] for index in range(args.questions)]
    unavailable = [
        q.task for q in questions if model.capabilities[q.task or ""].status == "unavailable"
    ]
    if unavailable:
        parser.error(
            f"Benchmark cards unavailable in this revision: {', '.join(map(str, unavailable))}"
        )
    report = benchmark(model, audio, questions, args.warmup, args.hops, args.per_question)
    torch = importlib.import_module("torch") if model.runtime.backend == "torch" else None
    head = getattr(getattr(model, "_engine", None), "head", None)
    report.update(
        {
            "format_version": 1,
            "runtime": model.runtime.to_dict(),
            "revision": getattr(model, "revision", None),
            "synthetic_model": args.fake,
            "versions": {
                "marimba": package_version("marimba"),
                "torch": str(torch.__version__) if torch is not None else package_version("torch"),
                "transformers": package_version("transformers"),
                "python": platform.python_version(),
            },
            "hardware": {
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "torch_threads": torch.get_num_threads() if torch is not None else None,
                "gpu": torch.cuda.get_device_name()
                if torch is not None and model.runtime.device == "cuda"
                else None,
            },
            "head_dtype": str(next(head.model.parameters()).dtype) if head is not None else None,
            "requested_dtype": args.dtype,
            "protocol": {
                "warmup": args.warmup,
                "hops": args.hops,
                "window_seconds": 8.0,
                "hop_seconds": 0.25,
                "prefill_seconds": 8.0,
                "audio_replay": "cyclic, offline sequential, no pacing sleeps",
                "latency_scope": "synchronized model call; no ingestion/decoding",
                "encoder_share_instrumented": args.per_question and hasattr(model, "_engine"),
            },
            "audio": {
                "path": str(args.audio) if args.audio is not None else None,
                "kind": "file" if args.audio is not None else "synthetic silence",
                "sampling_rate": 16000,
                "samples": len(audio),
                "crop": "last eight seconds",
                "float32_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            },
            "questions": [asdict(question) for question in questions],
            "hops": args.hops,
            "p50_ms": report["full_set"]["p50_ms"],
            "p95_ms": report["full_set"]["p95_ms"],
        }
    )
    output = json.dumps(report, indent=2)
    if args.json_path is not None:
        args.json_path.write_text(output + "\n")
    print(json.dumps(report))
    return 0
