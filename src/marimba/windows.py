"""Immutable, owned inference inputs and layer features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from .audio import SAMPLE_RATE, AudioArray, Mask, normalize_array

MAX_DECODER_SECONDS = 30.0
WINDOW_SAMPLES = 8 * SAMPLE_RATE
# Seconds of (reference, segment) each pair readout was trained on.
PAIR_LAYOUTS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {"same_speaker": (4.0, 4.0), "speaker_change": (6.0, 2.0)}
)


def readonly(array: npt.NDArray[Any]) -> Any:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class WindowSnapshot:
    audio: AudioArray
    mask: Mask
    window_start_s: float
    window_end_s: float

    def __post_init__(self) -> None:
        if (
            self.audio.ndim != 1
            or self.audio.dtype != np.float32
            or self.mask.shape != self.audio.shape
            or self.mask.dtype != np.bool_
            or not np.isfinite(self.audio).all()
            or not np.isfinite([self.window_start_s, self.window_end_s]).all()
            or self.window_start_s < 0
            or self.window_end_s < self.window_start_s
        ):
            raise ValueError("Invalid snapshot audio, mask, or timestamps")
        object.__setattr__(self, "audio", readonly(self.audio))
        object.__setattr__(self, "mask", readonly(self.mask))

    @property
    def seconds(self) -> float:
        return len(self.audio) / 16000


def decoder_snapshot(snapshot: WindowSnapshot) -> WindowSnapshot:
    first = int(np.argmax(snapshot.mask))
    if first == 0:
        return snapshot
    audio = np.zeros_like(snapshot.audio)
    mask = np.zeros_like(snapshot.mask)
    audio[:-first] = snapshot.audio[first:]
    mask[:-first] = snapshot.mask[first:]
    return WindowSnapshot(audio, mask, snapshot.window_start_s, snapshot.window_end_s)


def _fill(part: AudioArray, size: int, *, keep_end: bool) -> AudioArray:
    if len(part) >= size:
        return part[-size:] if keep_end else part[:size]
    # Pair readouts mean-pool each part: repetition keeps that mean, silence would dilute it.
    return np.resize(part, size)


def assemble_pair(task: str, reference: Any, segment: Any) -> AudioArray:
    """Build the eight-second ``[reference | segment]`` window a pair question expects."""
    layout = PAIR_LAYOUTS.get(task)
    if layout is None:
        raise ValueError(f"Pair windows exist for {', '.join(PAIR_LAYOUTS)}; got {task!r}")
    parts = [normalize_array(reference), normalize_array(segment)]
    if not all(len(part) for part in parts):
        raise ValueError("Reference and segment must both contain 16 kHz mono audio")
    sizes = [round(seconds * SAMPLE_RATE) for seconds in layout]
    return np.concatenate(
        (_fill(parts[0], sizes[0], keep_end=True), _fill(parts[1], sizes[1], keep_end=False))
    )


def full_window(audio: Any) -> WindowSnapshot:
    """Wrap exactly eight seconds of 16 kHz mono audio as a fully valid snapshot."""
    window = normalize_array(audio)
    if window.shape != (WINDOW_SAMPLES,):
        raise ValueError(f"An assembled window needs exactly {WINDOW_SAMPLES} samples at 16 kHz")
    return WindowSnapshot(window, np.ones(WINDOW_SAMPLES, dtype=np.bool_), 0.0, 8.0)


@dataclass(frozen=True)
class EncodedWindow:
    snapshot: WindowSnapshot
    features: Mapping[int, npt.NDArray[Any]] = field(default_factory=dict)
    encoder_latency_ms: float = 0.0
    audio_features: npt.NDArray[Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            not 0 < self.snapshot.seconds <= MAX_DECODER_SECONDS
            or not np.isfinite(self.encoder_latency_ms)
            or self.encoder_latency_ms < 0
        ):
            raise ValueError("Invalid encoded duration or encoder latency")
        object.__setattr__(
            self,
            "features",
            MappingProxyType({layer: readonly(value) for layer, value in self.features.items()}),
        )
        if self.audio_features is not None:
            object.__setattr__(self, "audio_features", readonly(self.audio_features))

    @property
    def l9(self) -> npt.NDArray[Any] | None:
        return self.features.get(9)

    @property
    def l18(self) -> npt.NDArray[Any] | None:
        return self.features.get(18)
