"""Runtime diagnostics and synchronized per-hop benchmarks."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .model import Marimba
from .questions import Question
from .runtime import detect_runtime, select_backend
from .testing import FakeMarimba


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marimba")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    bench = commands.add_parser("bench")
    bench.add_argument("--questions", type=int, default=1)
    bench.add_argument("--hops", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--backend", default="auto")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--fake", action="store_true")
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
    model = (
        FakeMarimba()
        if args.fake
        else Marimba.from_pretrained(backend=args.backend, device=args.device)
    )
    stream = model.stream([Question.noul("Is someone speaking?")] * args.questions)
    chunk = np.zeros(4000, dtype=np.float32)
    timings = []
    for index in range(args.warmup + args.hops):
        update = stream.push(chunk)[0]
        if index >= args.warmup:
            timings.append(update.latency_ms)
    print(
        json.dumps(
            {
                "runtime": model.runtime.to_dict(),
                "hops": len(timings),
                "p50_ms": float(np.percentile(timings, 50)),
                "p95_ms": float(np.percentile(timings, 95)),
            }
        )
    )
    return 0
