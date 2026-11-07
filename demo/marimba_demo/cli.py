"""Launch the demo with one local command."""

import argparse
import json
import os
import subprocess
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .scheduler import cadence_settings, review_context
from .server import create_app


def parse_intervals(value: str) -> dict[str, float]:
    try:
        pairs = [item.split("=") for item in value.split(",")]
        values = {task: float(interval) for task, interval in pairs}
        if len(values) != len(pairs):
            raise ValueError("Duplicate tasks.")
        return cadence_settings(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use task=seconds pairs, each at least 0.25 seconds."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="charcot-ai/marimba")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--review-window",
        type=review_context,
        default=8.0,
        help="Buffered review context, 8–30 seconds; above 8 is unvalidated",
    )
    parser.add_argument(
        "--review-intervals",
        type=parse_intervals,
        help="Per-card minimum pause after completion, e.g. language=2,term=4",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated supported tasks; defaults to all available cards",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    static = root / "marimba_demo" / "static"
    local = root / "local"
    frontend = root / "frontend"
    if not (static / "index.html").exists():
        if not frontend.exists():
            parser.error(
                "Frontend assets missing. Build with npm ci && npm run build in demo/frontend before installing."
            )
        subprocess.run(
            ["npm", "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True
        )
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)
    if (local / "stt_adapter.ts").is_file():
        subprocess.run(
            [
                str(frontend / "node_modules" / ".bin" / "esbuild"),
                str(local / "stt_adapter.ts"),
                "--bundle",
                "--format=esm",
                f"--outfile={local / 'stt_adapter.js'}",
            ],
            check=True,
        )
    try:
        from marimba import Marimba, Question
    except ImportError:
        parser.error(
            "Install the inference package in this environment first: uv pip install -e ."
        )
    print("Loading model. Audio stays in memory; no audio is recorded by this server.")
    model = Marimba.from_pretrained(
        args.model, revision=args.revision, device=args.device, backend=args.backend
    )
    app = create_app(
        model,
        Question,
        static=static,
        local=local,
        port=args.port,
        stt_config=json.loads(os.environ.get("MARIMBA_STT_CONFIG", "{}")),
        tasks=set(args.tasks.split(",")) if args.tasks else None,
        review_seconds=args.review_window,
        review_intervals=args.review_intervals,
    )
    if not args.no_open:
        threading.Timer(
            1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")
        ).start()
    uvicorn.run(
        app, host="127.0.0.1", port=args.port, ws_max_size=16384, ws_max_queue=1
    )
