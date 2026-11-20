"""Compare the batched encoder forward with the sequential path on public speech.

Windows are consecutive eight-second cuts of the concatenated validation utterances of a
public LibriSpeech sample, so some windows cross a speaker boundary. Exits nonzero when a
probability moves by more than the tolerance or a default-threshold decision flips.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from marimba import Marimba, Question
from marimba.audio import decode_audio

DATASET = "hf-internal-testing/librispeech_asr_dummy"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
QUESTIONS = [
    Question.score("Has the speaker finished?"),
    Question.noul("Is someone speaking?"),
    Question.noul("Is someone speaking?", seconds=1.0, threshold=0.1),
    Question.score("Is this the same speaker?"),
    Question.score("Did the speaker change?"),
    Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
]


def public_windows(limit: int) -> list[Any]:
    from datasets import Audio, load_dataset

    rows = load_dataset(DATASET, "clean", split="validation", revision=DATASET_REVISION)
    rows = rows.cast_column("audio", Audio(decode=False))
    audio = np.concatenate([decode_audio(row["audio"]) for row in rows])
    count = min(limit, len(audio) // 128000)
    return [audio[index * 128000 : (index + 1) * 128000] for index in range(count)]


def label(question: Question) -> str:
    return f"{question.task}@{question.seconds:g}s" if question.seconds else str(question.task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    model = Marimba.from_pretrained(device=args.device, dtype=args.dtype)
    windows = public_windows(args.windows)
    sequential = model.decide_windows(windows, QUESTIONS)
    batched = model.decide_windows(windows, QUESTIONS, batched=True, batch_size=args.batch_size)
    report: dict[str, Any] = {
        "dataset": DATASET,
        "runtime": model.runtime.to_dict(),
        "windows": len(windows),
        "batch_size": args.batch_size,
        "tolerance": args.tolerance,
        "tasks": {},
    }
    passed = True
    for index, question in enumerate(QUESTIONS):
        left = [row[index] for row in sequential]
        right = [row[index] for row in batched]
        deltas = np.array(
            [
                max(abs(a.probabilities[key] - b.probabilities[key]) for key in a.probabilities)
                if a.probabilities
                else abs((a.probability or 0.0) - (b.probability or 0.0))
                for a, b in zip(left, right, strict=True)
            ]
        )
        flips = sum(
            (a.probability >= a.threshold) != (b.probability >= b.threshold)
            if a.probabilities is None
            else a.value != b.value
            for a, b in zip(left, right, strict=True)
        )
        report["tasks"][label(question)] = {
            "max_abs_delta": float(deltas.max()),
            "mean_abs_delta": float(deltas.mean()),
            "decision_flips": int(flips),
        }
        passed &= bool(deltas.max() <= args.tolerance) and not flips
    report["passed"] = passed
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
