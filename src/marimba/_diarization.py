"""NumPy features and calibrated readouts for full eight-second windows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
TASKS = ("same_speaker", "speaker_change", "speaker_count")


def frame_centers_seconds(frame_count: int) -> Array:
    if frame_count <= 0:
        raise ValueError("Frame count must be positive")
    indices = np.minimum(
        ((np.arange(frame_count) + 0.5) * 128000 / frame_count).astype(int), 127999
    )
    return np.asarray(indices.astype(np.float64) / 16000)


def pool_interval(frames: Array, centers_seconds: Array, interval: tuple[float, float]) -> Array:
    values = np.asarray(frames, dtype=np.float32)
    centers = np.asarray(centers_seconds, dtype=np.float64)
    start, end = interval
    if (
        values.ndim != 2
        or centers.ndim != 1
        or values.shape[0] != len(centers)
        or not np.isfinite(values).all()
        or not np.isfinite(centers).all()
        or end < start
    ):
        raise ValueError("Expected finite frames, aligned centers and an ordered interval")
    selected = values[(centers >= start) & (centers < end)]
    if not len(selected):
        raise ValueError(f"No frame centers in interval {interval}")
    return selected.mean(axis=0, dtype=np.float32)


def build_same_speaker_features(embedding_a: Array, embedding_b: Array) -> Array:
    a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
    b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)
    if not len(a) or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected matching nonempty finite embeddings")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Speaker embeddings cannot have a zero norm")
    a, b = a / norm_a, b / norm_b
    return np.concatenate((a * b, np.abs(a - b))).astype(np.float32)


def task_features(task: str, frames: Array) -> Array:
    values = np.asarray(frames, dtype=np.float32)
    if (
        values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected nonempty finite [time, dimension] frames")
    centers = frame_centers_seconds(len(values))
    if task == "same_speaker":
        return build_same_speaker_features(
            pool_interval(values, centers, (0, 4)), pool_interval(values, centers, (4, 8))
        )
    if task == "speaker_change":
        prefix = pool_interval(values, centers, (0, 6))
        last = pool_interval(values, centers, (6, 8))
        return np.concatenate((build_same_speaker_features(prefix, last), prefix, last))
    if task == "speaker_count":
        return np.concatenate(
            (values.mean(axis=0, dtype=np.float32), values.std(axis=0, dtype=np.float32))
        )
    raise ValueError(f"Unknown diarization task: {task}")


def _sigmoid(values: Array) -> Array:
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_x = np.exp(values[~positive])
    output[~positive] = exp_x / (1 + exp_x)
    return output


class DiarizationReadout:
    def __init__(self, task: str, weights: Mapping[str, Array]) -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown diarization task: {task}")
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in weights.values()):
            raise ValueError("Readout tensors must be finite float32")
        self.task = task
        self.weights = weights

    def predict(self, features: Array) -> Array:
        x = np.array(features, dtype=np.float32, copy=True)
        dimension = 3584 if self.task == "speaker_change" else 1792
        if x.ndim != 2 or x.shape[1] != dimension or not np.isfinite(x).all():
            raise ValueError(f"Expected finite [batch, {dimension}] task features")
        w = self.weights
        if self.task != "speaker_change":
            x -= w["scaler_mean"]
            x /= w["scaler_scale"]
            logits = x.astype(np.float64) @ w["coef"].astype(np.float64).T
            logits += w["intercept"].astype(np.float64)
        else:
            hidden = np.maximum(x @ w["layer_0_weight"] + w["layer_0_bias"], 0)
            z = (hidden @ w["layer_1_weight"] + w["layer_1_bias"])[:, 0]
            p1 = _sigmoid(z)
            logits = np.log(np.clip(p1, 1e-12, 1)) - np.log(np.clip(1 - p1, 1e-12, 1))
        if self.task == "speaker_count":
            scores = logits / float(w["temperature"][0])
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            return np.asarray(probs / probs.sum(axis=1, keepdims=True))
        positive = _sigmoid(
            logits.reshape(-1) * w["platt_coefficient"][0] + w["platt_intercept"][0]
        )
        return np.column_stack((1 - positive, positive))
