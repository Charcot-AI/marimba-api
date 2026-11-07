"""Shared model interface, independent of any inference engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from threading import RLock
from time import perf_counter
from typing import Any, Literal

from .audio import SAMPLE_RATE, AudioArray, Mask, decode_audio, right_aligned_window
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
