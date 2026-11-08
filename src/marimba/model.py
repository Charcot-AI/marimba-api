"""Shared model interface, independent of any inference engine."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal

from .audio import SAMPLE_RATE, AudioArray, Mask, decode_audio, right_aligned_window
from .calibration import Calibration, fit_platt
from .capabilities import DEFAULT_CAPABILITIES, Capability
from .decision import Decision
from .questions import Question, UnsupportedQuestion
from .runtime import RuntimeInfo, select_backend, synchronize
from .stream import Stream
from .windows import MAX_DECODER_SECONDS, EncodedWindow, WindowSnapshot


class Marimba:
    def __init__(self, runtime: RuntimeInfo | None = None) -> None:
        self.runtime = runtime if runtime is not None else select_backend()
        self._encoder_lock = RLock()
        self._decoder_lock = RLock()
        self._calibrations: dict[tuple[str, str], Calibration] = {}
        self._calibration_lock = RLock()
        self._fitting: ContextVar[bool] = ContextVar("fitting_calibration", default=False)

    def lane(self, question: Question) -> Literal["encoder", "decoder"]:
        if question.task not in self.capabilities:
            raise UnsupportedQuestion(f"Unsupported task {question.task!r}")
        return "decoder" if question.task in ("language", "term") else "encoder"

    def _encode_snapshot(self, snapshot: WindowSnapshot) -> EncodedWindow:
        raise NotImplementedError("Implement snapshot encoding in a model backend")

    def encode(self, snapshot: WindowSnapshot) -> EncodedWindow:
        if len(snapshot.audio) != 128000:
            raise ValueError("Encoder lanes require exactly an eight-second snapshot")
        return self._encode_snapshot(snapshot)

    def decide_encoded(
        self, encoded: EncodedWindow, questions: Sequence[Question]
    ) -> list[Decision]:
        raise NotImplementedError("Implement encoded decisions in a model backend")

    def decide_snapshot(
        self, snapshot: WindowSnapshot, questions: Sequence[Question]
    ) -> list[Decision]:
        if any(self.lane(question) != "decoder" for question in questions):
            raise ValueError("decide_snapshot accepts decoder questions only")
        if not 0 < snapshot.seconds <= MAX_DECODER_SECONDS or not snapshot.mask.any():
            raise ValueError(
                f"Decoder snapshots must contain audio and be <= {MAX_DECODER_SECONDS} seconds"
            )
        if not questions:
            return []
        return self.decide_encoded(self._encode_snapshot(snapshot), questions)

    def calibrate(
        self,
        audio: Sequence[Any],
        labels: Sequence[bool | int],
        question: Question,
        *,
        method: str = "platt",
        batch_size: int = 16,
    ) -> Calibration:
        if question.kind == "choice" or method != "platt":
            raise ValueError("Platt calibration supports noul/score questions only")
        if batch_size <= 0 or len(audio) != len(labels) or len(labels) == 0:
            raise ValueError("Audio and labels must have equal nonzero lengths; batch_size > 0")
        if any(label not in (0, 1) for label in labels) or not 0 < sum(labels) < len(labels):
            raise ValueError("Calibration requires both positive and negative 0/1 labels")
        if len(labels) < 100:
            warnings.warn(
                "Calibration has fewer than 100 labeled windows", UserWarning, stacklevel=2
            )
        token = self._fitting.set(True)
        try:
            rows = self.batch(audio, [question], batch_size=batch_size)
        finally:
            self._fitting.reset(token)
        probabilities = [row[0].probability for row in rows]
        if any(probability is None for probability in probabilities):
            raise ValueError("Cannot calibrate a lane without probabilities")
        a, b = fit_platt(probabilities, labels)
        calibration = Calibration(
            question.text,
            question.kind,
            method,
            a,
            b,
            len(labels),
            int(sum(labels)),
            getattr(self, "revision", None),
        )
        with self._calibration_lock:
            self._calibrations[(question.text, question.kind)] = calibration
        return calibration

    def _apply_calibrations(self, decisions: list[Decision]) -> list[Decision]:
        if self._fitting.get():
            return decisions
        with self._calibration_lock:
            calibrations = dict(self._calibrations)
        results = []
        for decision in decisions:
            calibration = calibrations.get((decision.question.text, decision.kind))
            if calibration is not None and decision.probability is not None:
                probability = calibration.apply(decision.probability)
                value = (
                    probability >= decision.threshold if decision.kind == "noul" else probability
                )
                decision = replace(decision, probability=probability, value=value, calibrated=True)
            results.append(decision)
        return results

    def save_calibrations(self, path: str | Path) -> None:
        with self._calibration_lock:
            records = [calibration.to_dict() for calibration in self._calibrations.values()]
        Path(path).write_text(json.dumps({"format_version": 1, "calibrations": records}, indent=2))

    def load_calibrations(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("format_version") != 1:
            raise ValueError("Unsupported calibration file")
        if not isinstance(data.get("calibrations"), list):
            raise ValueError("Invalid calibration records")
        records = [Calibration.from_dict(record) for record in data["calibrations"]]
        with self._calibration_lock:
            self._calibrations.update(
                {(record.question, record.kind): record for record in records}
            )

    def clear_calibration(self, question: Question) -> None:
        with self._calibration_lock:
            self._calibrations.pop((question.text, question.kind), None)

    @classmethod
    def from_pretrained(
        cls,
        repo: str = "charcot-ai/marimba",
        revision: str | None = None,
        device: str = "auto",
        backend: str = "auto",
        dtype: str = "auto",
    ) -> Marimba:
        from .pretrained import load_pretrained

        return load_pretrained(repo, revision, device, backend, dtype)

    @property
    def capabilities(self) -> dict[str, Capability]:
        return dict(DEFAULT_CAPABILITIES)

    def decide_window(
        self,
        window: AudioArray,
        mask: Mask,
        questions: list[Question],
        start_s: float,
        end_s: float,
    ) -> list[Decision]:
        raise NotImplementedError("Implement shared-window inference in a model backend")

    def __call__(
        self, audio: Any, questions: Sequence[Question], sampling_rate: int | None = None
    ) -> list[Decision]:
        samples = decode_audio(audio, sampling_rate)
        window, mask = right_aligned_window(samples)
        end = len(samples) / SAMPLE_RATE
        synchronize(self.runtime)
        started = perf_counter()
        decisions = self.decide_window(window, mask, list(questions), max(0.0, end - 8), end)
        synchronize(self.runtime)
        latency = (perf_counter() - started) * 1000
        return [replace(d, latency_ms=latency) for d in decisions]

    def batch(
        self, audios: Sequence[Any], questions: Sequence[Question], batch_size: int = 16
    ) -> list[list[Decision]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return [self(audio, questions) for audio in audios]

    def map_fn(
        self, questions: Sequence[Question], audio_column: str = "audio", prefix: str = "marimba_"
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        frozen_questions = list(questions)

        def apply(batch: dict[str, Any]) -> dict[str, Any]:
            results = self.batch(batch[audio_column], frozen_questions)
            return {
                f"{prefix}{index}": [row[index].to_dict() for row in results]
                for index in range(len(frozen_questions))
            }

        return apply

    def stream(
        self,
        questions: Sequence[Question],
        window_seconds: float = 8.0,
        hop_seconds: float = 0.25,
        sample_rate: int = 16000,
        coalesce: bool = False,
        context_seconds: float = 8.0,
        decoder_hop_seconds: float | None = None,
        decoder_context_seconds: float | None = None,
    ) -> Stream:
        return Stream(
            self,
            list(questions),
            window_seconds,
            hop_seconds,
            sample_rate,
            coalesce,
            context_seconds,
            decoder_hop_seconds,
            decoder_context_seconds,
        )
