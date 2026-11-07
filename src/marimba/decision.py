"""JSON-safe inference results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .questions import Kind, Question


@dataclass(frozen=True)
class Decision:
    question: Question
    task: str
    kind: Kind
    value: bool | float | str | None
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    score: float | None = None
    threshold: float = 0.5
    window_start_s: float = 0.0
    window_end_s: float = 0.0
    latency_ms: float = 0.0
    status: str = "unavailable"
    calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.question.options is not None:
            result["question"]["options"] = list(self.question.options)
        return result


@dataclass(frozen=True)
class StreamUpdate:
    window_start_s: float
    window_end_s: float
    decisions: list[Decision]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "latency_ms": self.latency_ms,
        }
