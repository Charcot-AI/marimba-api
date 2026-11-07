"""Deterministic synthetic model for integration development, not real inference."""

from __future__ import annotations

import hashlib

import numpy as np

from .audio import AudioArray, Mask
from .capabilities import Capability
from .decision import Decision
from .model import Marimba
from .questions import Question
from .runtime import RuntimeInfo


class FakeMarimba(Marimba):
    def __init__(self) -> None:
        super().__init__(RuntimeInfo("fake", "cpu", "float32"))

    @property
    def capabilities(self) -> dict[str, Capability]:
        return {
            task: Capability(
                task,
                "experimental",
                "synthetic",
                cap.kinds,
                "Synthetic decisions only; not inference",
            )
            for task, cap in super().capabilities.items()
        }

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        energy = float(np.mean(window[mask] ** 2)) if mask.any() else 0.0
        decisions = []
        for question in questions:
            offset = (
                int.from_bytes(hashlib.sha256(question.text.encode()).digest()[:4], "big") / 2**32
            )
            probability = float(min(1.0, energy * 8 + offset * 0.1))
            probabilities = None
            value: bool | float | str = probability
            if question.kind == "noul":
                value = probability >= question.threshold
            elif question.kind == "choice":
                options = question.options or ()
                weights = np.arange(1, len(options) + 1, dtype=np.float64) + probability
                weights /= weights.sum()
                probabilities = dict(zip(options, map(float, weights), strict=True))
                value = max(probabilities, key=lambda key: probabilities[key])
                probability = probabilities[value]
            decisions.append(
                Decision(
                    question,
                    question.task or "",
                    question.kind,
                    value,
                    probability,
                    probabilities,
                    probability,
                    question.threshold,
                    start_s,
                    end_s,
                    status="experimental",
                )
            )
        return decisions
