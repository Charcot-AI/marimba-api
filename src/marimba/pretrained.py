"""Inference dispatch for manifest-backed torch model lanes."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any

import numpy as np

from .audio import AudioArray, Mask
from .capabilities import Capability
from .decision import Decision
from .language import LANGUAGES
from .lexical import normalize
from .model import Marimba
from .questions import Question, UnsupportedQuestion
from .runtime import RuntimeInfo, select_backend
from .weights import load_manifest


class PretrainedMarimba(Marimba):
    def __init__(self, runtime: RuntimeInfo, engine: Any, manifest: dict[str, Any]) -> None:
        super().__init__(runtime)
        self._engine = engine
        self.revision: str = manifest["resolved_revision"]
        self._lock = RLock()
        self._capabilities = {
            task: replace(
                cap, status="unavailable", evaluation="No weights installed for this lane"
            )
            for task, cap in super().capabilities.items()
        }
        for task in ("turn_end", "language", "term"):
            lane = manifest["lanes"].get(task, {})
            if lane.get("status") not in ("validated", "experimental"):
                continue
            cap = super().capabilities[task]
            self._capabilities[task] = replace(
                cap, status=lane["status"], evaluation=lane.get("evaluation", cap.evaluation)
            )
        if manifest["lanes"].get("term", {}).get("calibration") != "none":
            raise ValueError("Only explicitly uncalibrated lexical manifests are supported")

    @property
    def capabilities(self) -> dict[str, Capability]:
        return dict(self._capabilities)

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        for question in questions:
            cap = self._capabilities.get(question.task or "")
            if cap is None:
                raise UnsupportedQuestion(f"Unsupported task {question.task!r}")
            if question.kind not in cap.kinds:
                raise UnsupportedQuestion(f"Task {question.task} supports kinds {cap.kinds}")
            if question.task == "term" and (not question.term or not normalize(question.term)):
                raise UnsupportedQuestion("Term questions require a nonempty quoted term")
            if question.task == "language":
                unsupported = set(question.options or ()) - LANGUAGES.keys()
                if unsupported:
                    raise ValueError(
                        f"Unsupported language option(s): {', '.join(sorted(unsupported))}"
                    )
        if not questions:
            return []
        if window.shape != (128000,) or mask.shape != window.shape or not mask.any():
            raise ValueError(
                "Expected a nonempty right-aligned eight-second window with a valid mask"
            )
        with self._lock:
            return self._decide(window, mask, questions, start_s, end_s)

    def _decide(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        active = {
            q.task for q in questions if self._capabilities[q.task or ""].status != "unavailable"
        }
        encoded = self._engine.encode(window) if active else None
        turn = self._engine.turn_end(mask) if "turn_end" in active else None
        options = sorted(
            {option for q in questions if q.task == "language" for option in q.options or ()}
        )
        language = self._engine.language_scores(encoded, options) if "language" in active else {}
        hypotheses = self._engine.hypotheses(encoded) if "term" in active else []
        decisions = []
        for question in questions:
            task = question.task or ""
            cap = self._capabilities[task]
            value: bool | float | str | None = None
            probability = score = None
            probabilities = None
            if cap.status != "unavailable":
                if task == "turn_end" and turn is not None:
                    score, probability = map(float, turn)
                    value = probability
                elif task == "language":
                    names = question.options or ()
                    scores = np.array([language[name] for name in names], dtype=np.float64)
                    weights = np.exp(scores - np.max(scores))
                    weights /= weights.sum()
                    probabilities = dict(zip(names, map(float, weights), strict=True))
                    value = names[int(np.argmax(weights))]
                    probability = probabilities[value]
                elif task == "term":
                    probability = score = float(self._engine.term_score(question.term, hypotheses))
                    value = probability >= question.threshold
            decisions.append(
                Decision(
                    question,
                    task,
                    question.kind,
                    value,
                    probability,
                    probabilities,
                    score,
                    question.threshold,
                    start_s,
                    end_s,
                    status=cap.status,
                )
            )
        return decisions


def load_pretrained(
    repo: str, revision: str | None, device: str, backend: str, dtype: str
) -> Marimba:
    if backend not in ("auto", "torch"):
        raise ValueError("These model lanes require backend='torch'; ONNX is not available yet")
    runtime = select_backend("torch", device, dtype)
    manifest, paths = load_manifest(repo, revision)
    try:
        from ._qwen import QwenEngine
    except ImportError as error:
        raise ImportError("Install marimba[torch] to load Qwen3-ASR model lanes") from error
    return PretrainedMarimba(runtime, QwenEngine(runtime, paths), manifest)
