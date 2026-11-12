"""Smoke-test each typed question and the synthetic examples through Gradio."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

from gradio_client import Client, handle_file

QUESTIONS = {
    "Speech": "voice_activity",
    "Turn ended": "turn_end",
    "Language": "language",
    "Term": "term",
    "Speaker change": "speaker_change",
    "People": "speaker_count",
}
EXAMPLES = Path(__file__).parent / "examples"


def check_decisions(decisions: list[dict], selected: list[str]) -> None:
    assert [item["task"] for item in decisions] == [QUESTIONS[name] for name in selected]
    for item in decisions:
        assert item["status"] in ("validated", "experimental"), item
        assert item["value"] is not None, item
        probability = item["probability"]
        assert probability is not None and math.isfinite(probability) and 0 <= probability <= 1
        if item["probabilities"]:
            assert math.isclose(sum(item["probabilities"].values()), 1, abs_tol=1e-5)
        if item["task"] == "term":
            assert not item["calibrated"]
            assert math.isclose(probability * 5, round(probability * 5), abs_tol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="charcot-ai/marimba")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    client = Client(args.repo_id, token=os.environ["HF_TOKEN"], verbose=False)
    cases = [("03_term", [name]) for name in QUESTIONS]
    cases.extend((path.stem, list(QUESTIONS)) for path in sorted(EXAMPLES.glob("*.wav")))
    results = []
    for name, selected in cases:
        started = perf_counter()
        decisions = client.predict(
            handle_file(str(EXAMPLES / f"{name}.wav")), selected, "marimba", api_name="/decide"
        )
        wall_s = perf_counter() - started
        check_decisions(decisions, selected)
        result = {"example": name, "questions": selected, "wall_s": wall_s, "decisions": decisions}
        results.append(result)
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
