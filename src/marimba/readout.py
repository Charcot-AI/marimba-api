"""Native-frame tail pooling and calibrated linear readout."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .audio import SAMPLE_RATE, Mask


def pool_layer(frames: npt.NDArray[np.float32], sample_mask: Mask) -> npt.NDArray[np.float32]:
    size = 8 * SAMPLE_RATE
    if frames.ndim != 2 or not len(frames) or sample_mask.shape != (size,):
        raise ValueError("Expected native frames and an eight-second sample mask")
    centers = np.minimum(
        ((np.arange(len(frames)) + 0.5) * size / len(frames)).astype(int), size - 1
    )
    tail = sample_mask[centers] & (centers >= size - 2 * SAMPLE_RATE)
    if not tail.any():
        raise ValueError("No valid end-of-window frames")
    return np.asarray(frames[tail].mean(axis=0, dtype=np.float32), dtype=np.float32)


def sigmoid(value: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -value)))


def predict_readout(
    weights: Mapping[str, npt.NDArray[np.float64]], features: npt.NDArray[np.float32]
) -> tuple[float, float]:
    logit = float(
        ((features - weights["mean"]) / weights["scale"]) @ weights["coef"]
        + weights["intercept"][0]
    )
    return sigmoid(logit), sigmoid(
        logit * float(weights["platt_coef"][0]) + float(weights["platt_intercept"][0])
    )
