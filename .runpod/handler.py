"""Runpod Serverless worker for Language and Term decoder questions."""

from __future__ import annotations

import base64
import logging
import time
from functools import cache
from typing import Any

import numpy as np

from marimba import Marimba, Question, WindowSnapshot
from marimba.windows import MAX_DECODER_SECONDS

logger = logging.getLogger("marimba-runpod")


@cache
def _load_model() -> Marimba:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the worker container")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logger.info("Loading Marimba decoder model on CUDA")
    model = Marimba.from_pretrained(device="cuda", backend="torch")
    logger.info("Marimba decoder model is ready")
    return model


def _snapshot(data: dict[str, Any]) -> WindowSnapshot:
    if data.get("sample_rate") != 16000:
        raise ValueError("Expected 16000 Hz PCM")
    audio_bytes = base64.b64decode(data["pcm16_b64"], validate=True)
    mask_bytes = base64.b64decode(data["mask_b64"], validate=True)
    if len(audio_bytes) % 2:
        raise ValueError("PCM16 payload has an odd byte count")

    pcm = np.frombuffer(audio_bytes, dtype="<i2")
    mask = np.frombuffer(mask_bytes, dtype=np.uint8)
    if len(pcm) != len(mask):
        raise ValueError("PCM and mask lengths differ")
    if not 0 < len(pcm) <= 16000 * MAX_DECODER_SECONDS or not mask.any():
        raise ValueError("Decoder snapshots require valid audio and at most 30 seconds")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask bytes must be zero or one")

    return WindowSnapshot(
        pcm.astype(np.float32) / 32768.0,
        mask.astype(np.bool_),
        float(data["window_start_s"]),
        float(data["window_end_s"]),
    )


def _questions(values: list[dict[str, Any]]) -> list[Question]:
    result = []
    for value in values:
        question = dict(value)
        if question.get("options") is not None:
            question["options"] = tuple(question["options"])
        parsed = Question(**question)
        if parsed.task not in ("language", "term"):
            raise ValueError("The remote worker accepts Language and Term questions only")
        result.append(parsed)
    if not result:
        raise ValueError("At least one decoder question is required")
    return result


def handler(job: dict[str, Any]) -> dict[str, Any]:
    data = job["input"]
    if data.get("lane") != "decoder":
        raise ValueError("The remote worker accepts decoder-lane requests only")

    snapshot = _snapshot(data)
    questions = _questions(data["questions"])
    model = _load_model()
    if any(model.lane(question) != "decoder" for question in questions):
        raise ValueError("The remote worker accepts decoder questions only")

    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    decisions = model.decide_snapshot(snapshot, questions)
    torch.cuda.synchronize()
    compute_ms = (time.perf_counter() - started) * 1000

    return {
        "decisions": [decision.to_dict() for decision in decisions],
        "gpu_compute_ms": compute_ms,
    }


def main() -> None:
    import runpod

    logging.basicConfig(level=logging.INFO)
    _load_model()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
