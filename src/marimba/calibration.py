"""Portable, positive-slope Platt calibration without optional dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .questions import Kind


def logit(probability: Any) -> Any:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True)
class Calibration:
    question: str
    kind: Kind
    method: str
    a: float
    b: float
    n: int
    n_positive: int
    created_with: str | None

    def __post_init__(self) -> None:
        if (
            not self.question.strip()
            or self.kind not in ("noul", "score")
            or self.method != "platt"
            or not np.isfinite([self.a, self.b]).all()
            or self.a <= 0
            or type(self.n) is not int
            or type(self.n_positive) is not int
            or not 0 < self.n_positive < self.n
            or (self.created_with is not None and not isinstance(self.created_with, str))
        ):
            raise ValueError("Invalid Platt calibration")

    def apply(self, probability: float) -> float:
        return float(np.exp(-np.logaddexp(0.0, -(self.a * logit(probability) + self.b))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        if not isinstance(data, dict):
            raise ValueError("Invalid calibration record")
        try:
            return cls(**data)
        except (TypeError, AttributeError) as error:
            raise ValueError("Invalid calibration record") from error


def fit_platt(probabilities: Any, labels: Any) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
        or not np.isin(labels, [0, 1]).all()
        or not 0 < np.sum(labels) < len(labels)
    ):
        raise ValueError(
            "Calibration requires finite probabilities and both positive/negative labels"
        )
    design = np.column_stack((logit(probabilities), np.ones(len(labels))))
    target = labels.astype(np.float64)
    coefficients = np.array([1.0, 0.0])
    penalty = np.array([1e-6, 0.0])

    def objective(parameters: Any) -> float:
        logits = design @ parameters
        return float(
            np.sum(np.logaddexp(0, logits) - target * logits)
            + 0.5 * np.sum(penalty * parameters**2)
        )

    for _ in range(100):
        logits = design @ coefficients
        predictions = np.exp(-np.logaddexp(0, -logits))
        gradient = design.T @ (predictions - target) + penalty * coefficients
        curvature = predictions * (1 - predictions)
        hessian = design.T @ (curvature[:, None] * design) + np.diag(penalty + 1e-10)
        step = np.linalg.solve(hessian, gradient)
        previous = objective(coefficients)
        rate = 1.0
        for _ in range(40):
            candidate = coefficients - rate * step
            candidate[0] = max(1e-8, candidate[0])
            if objective(candidate) <= previous:
                break
            rate *= 0.5
        else:
            break
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change < 1e-9:
            break
    return float(coefficients[0]), float(coefficients[1])
