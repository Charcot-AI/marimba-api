"""Deterministic synthetic model for integration development, not real inference."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from threading import Event
from time import perf_counter

import numpy as np

from .audio import AudioArray, Mask
from .capabilities import Capability
from .decision import Decision
from .model import Marimba
from .questions import Question
from .runtime import RuntimeInfo
from .windows import EncodedWindow, WindowSnapshot


class FakeMarimba(Marimba):
    def __init__(self, slow_decoder_seconds: float = 0.0) -> None:
        super().__init__(RuntimeInfo("fake", "cpu", "float32"))
        if not np.isfinite(slow_decoder_seconds) or slow_decoder_seconds < 0:
            raise ValueError("Slow decoder delay must be finite and nonnegative")
        self.slow_decoder_seconds = slow_decoder_seconds

    def _encode_snapshot(self, snapshot: WindowSnapshot) -> EncodedWindow:
        if not snapshot.mask.any():
            raise ValueError("Expected a nonempty snapshot")
        with self._encoder_lock:
            started = perf_counter()
            features = snapshot.audio[::1280, None]
            return EncodedWindow(
                snapshot, {9: features, 18: features}, (perf_counter() - started) * 1000
            )

    def decide_encoded(
        self, encoded: EncodedWindow, questions: Sequence[Question]
    ) -> list[Decision]:
        if any(self.lane(q) == "encoder" for q in questions) and encoded.snapshot.seconds != 8:
            raise ValueError("Encoder lanes require an eight-second snapshot")
        snapshot = encoded.snapshot
        results: dict[Question, Decision] = {}
        for lane in ("encoder", "decoder"):
            group = [q for q in questions if self.lane(q) == lane]
            if not group:
                continue

            def compute(group: list[Question] = group) -> list[Decision]:
                started = perf_counter()
                decisions = self.decide_window(
                    snapshot.audio,
                    snapshot.mask,
                    group,
                    snapshot.window_start_s,
                    snapshot.window_end_s,
                )
                latency = (perf_counter() - started) * 1000
                return [replace(d, latency_ms=latency) for d in decisions]

            if lane == "decoder":
                with self._decoder_lock:
                    decisions = compute()
            else:
                decisions = compute()
            results.update(zip(group, decisions, strict=True))
        return [results[q] for q in questions]

    @property
    def capabilities(self) -> dict[str, Capability]:
        return {
            task: cap
            if cap.status == "unavailable"
            else Capability(
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
        if any(self.lane(q) == "decoder" for q in questions) and self.slow_decoder_seconds:
            with self._decoder_lock:
                Event().wait(self.slow_decoder_seconds)
        energy = float(np.mean(window[mask] ** 2)) if mask.any() else 0.0
        decisions = []
        capabilities = self.capabilities
        for question in questions:
            cap = capabilities.get(question.task or "")
            if cap is not None and cap.status == "unavailable":
                decisions.append(
                    Decision(
                        question,
                        question.task or "",
                        question.kind,
                        None,
                        threshold=question.threshold,
                        window_start_s=start_s,
                        window_end_s=end_s,
                        status="unavailable",
                    )
                )
                continue
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
        return self._apply_calibrations(decisions)
