"""Inference dispatch for manifest-backed torch model lanes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

from .audio import AudioArray, Mask
from .capabilities import Capability
from .decision import Decision
from .language import LANGUAGES, option_probabilities, resolve_options
from .lexical import normalize
from .model import Marimba
from .pii import PIIDetector
from .questions import Question, UnsupportedQuestion, matches_template
from .readout import sigmoid
from .runtime import RuntimeInfo, select_backend, synchronize
from .weights import load_manifest
from .windows import EncodedWindow, WindowSnapshot

_CANONICAL_VAD_PROMPT = "Há fala humana neste áudio?"


class PretrainedMarimba(Marimba):
    def __init__(
        self,
        runtime: RuntimeInfo,
        engine: Any,
        manifest: dict[str, Any],
        pii_detector: PIIDetector | None = None,
    ) -> None:
        super().__init__(runtime)
        self._engine = engine
        self._pii_detector = pii_detector
        self.revision: str = manifest["resolved_revision"]
        self._capabilities = {
            task: replace(
                cap, status="unavailable", evaluation="No weights installed for this lane"
            )
            for task, cap in super().capabilities.items()
        }
        for task in ("turn_end", "language", "term", "voice_activity"):
            lane = manifest["lanes"].get(task, {})
            if lane.get("status") not in ("validated", "experimental"):
                continue
            cap = super().capabilities[task]
            self._capabilities[task] = replace(
                cap,
                status=lane["status"],
                evaluation=(
                    cap.evaluation
                    if task == "voice_activity"
                    else lane.get("evaluation", cap.evaluation)
                ),
            )
            if task in ("language", "term"):
                self._capabilities[task] = replace(
                    self._capabilities[task],
                    evaluation=(
                        self._capabilities[task].evaluation
                        + "; validated on eight-second windows only; longer context is unvalidated"
                    ),
                )
        pii_capability = super().capabilities["pii"]
        self._capabilities["pii"] = replace(
            pii_capability,
            status="experimental" if pii_detector is not None else "unavailable",
            evaluation=(
                "Your detector is unevaluated; an in-house pt-BR text PII model "
                "scored item F1 0.904 on synthetic speech"
                if pii_detector is not None
                else "Requires a pii_detector"
            ),
        )
        if manifest["lanes"].get("term", {}).get("calibration") != "none":
            raise ValueError("Only explicitly uncalibrated lexical manifests are supported")

    @property
    def capabilities(self) -> dict[str, Capability]:
        return dict(self._capabilities)

    def _validate(self, questions: Sequence[Question]) -> None:
        for question in questions:
            cap = self._capabilities.get(question.task or "")
            if cap is None:
                raise UnsupportedQuestion(f"Unsupported task {question.task!r}")
            if question.kind not in cap.kinds:
                raise UnsupportedQuestion(f"Task {question.task} supports kinds {cap.kinds}")
            if question.task == "term" and (not question.term or not normalize(question.term)):
                raise UnsupportedQuestion("Term questions require a nonempty quoted term")
            if question.task == "language":
                resolve_options(question.options or ())

    def _encode_snapshot(self, snapshot: WindowSnapshot) -> EncodedWindow:
        if not snapshot.mask.any():
            raise ValueError("Expected a nonempty snapshot")
        with self._encoder_lock:
            synchronize(self.runtime)
            started = perf_counter()
            data = self._engine.encode(snapshot.audio)
            synchronize(self.runtime)
            latency = (perf_counter() - started) * 1000
        return EncodedWindow(snapshot, data["features"], latency, data["audio_features"])

    def decide_snapshot(
        self, snapshot: WindowSnapshot, questions: Sequence[Question]
    ) -> list[Decision]:
        self._validate(questions)
        return super().decide_snapshot(snapshot, questions)

    def decide_encoded(
        self, encoded: EncodedWindow, questions: Sequence[Question]
    ) -> list[Decision]:
        self._validate(questions)
        if any(self.lane(q) == "encoder" for q in questions) and encoded.snapshot.seconds != 8:
            raise ValueError("Encoder lanes require an eight-second snapshot")
        results: dict[Question, Decision] = {}
        for lane in ("encoder", "decoder"):
            group = [q for q in questions if self.lane(q) == lane]
            if not group:
                continue
            if lane == "decoder":
                with self._decoder_lock:
                    decisions = self._timed_decide(encoded, group)
            else:
                decisions = self._timed_decide(encoded, group)
            results.update(zip(group, decisions, strict=True))
        return [results[q] for q in questions]

    def _timed_decide(self, encoded: EncodedWindow, questions: list[Question]) -> list[Decision]:
        synchronize(self.runtime)
        started = perf_counter()
        decisions = self._decide(encoded, questions)
        synchronize(self.runtime)
        latency = (perf_counter() - started) * 1000
        return [replace(d, latency_ms=latency) for d in decisions]

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        self._validate(questions)
        if not questions:
            return []
        if window.shape != (128000,) or mask.shape != window.shape or not mask.any():
            raise ValueError(
                "Expected a nonempty right-aligned eight-second window with a valid mask"
            )
        snapshot = WindowSnapshot(window, mask, start_s, end_s)
        active = any(self._capabilities[q.task or ""].status != "unavailable" for q in questions)
        encoded = self.encode(snapshot) if active else EncodedWindow(snapshot)
        return self.decide_encoded(encoded, questions)

    def _decide(
        self,
        encoded: EncodedWindow,
        questions: list[Question],
    ) -> list[Decision]:
        active = {
            q.task for q in questions if self._capabilities[q.task or ""].status != "unavailable"
        }
        snapshot = encoded.snapshot
        mask = snapshot.mask
        start_s, end_s = snapshot.window_start_s, snapshot.window_end_s
        features, audio = encoded.features, encoded.audio_features
        turn = self._engine.turn_end(features, mask) if "turn_end" in active else None
        resolved = {q: resolve_options(q.options or ()) for q in questions if q.task == "language"}
        options = sorted(
            {code for mapping in resolved.values() for code in mapping.values() if code is not None}
        )
        if any(None in mapping.values() for mapping in resolved.values()):
            options = sorted(LANGUAGES)
        prefill = getattr(self._engine, "prefill", None)
        prefix = (
            prefill(audio)
            if prefill is not None and "language" in active and {"term", "pii"} & active
            else None
        )
        language = (
            self._engine.language_scores(audio, options, prefix=prefix)
            if "language" in active
            else {}
        )
        hypotheses = self._engine.hypotheses(audio, prefix=prefix) if "term" in active else []
        pii_transcript = (
            hypotheses[0]
            if "term" in active
            else self._engine.transcript(audio, prefix=prefix)
            if "pii" in active
            else None
        )
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
                    probabilities = option_probabilities(language, resolved[question])
                    value = max(probabilities, key=lambda name: probabilities[name])
                    probability = probabilities[value]
                elif task == "term":
                    probability = score = float(self._engine.term_score(question.term, hypotheses))
                    value = probability >= question.threshold
                elif (
                    task == "pii" and pii_transcript is not None and self._pii_detector is not None
                ):
                    value = bool(self._pii_detector(pii_transcript))
                elif task == "voice_activity":
                    prompt = (
                        _CANONICAL_VAD_PROMPT
                        if matches_template("voice_activity", question.text)
                        else question.text
                    )
                    score = float(self._engine.head_score(features, mask, prompt))
                    probability = sigmoid(score)
                    value = (
                        probability
                        if question.kind == "score"
                        else probability >= question.threshold
                    )
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
        return self._apply_calibrations(decisions)


def load_pretrained(
    repo: str,
    revision: str | None,
    device: str,
    backend: str,
    dtype: str,
    pii_detector: PIIDetector | None = None,
) -> Marimba:
    if backend not in ("auto", "torch"):
        raise ValueError("These model lanes require backend='torch'; ONNX is not available yet")
    runtime = select_backend("torch", device, dtype)
    manifest, paths = load_manifest(repo, revision)
    try:
        from ._qwen import QwenEngine
    except ImportError as error:
        raise ImportError("Install marimba[torch] to load Qwen3-ASR model lanes") from error
    return PretrainedMarimba(runtime, QwenEngine(runtime, paths), manifest, pii_detector)
