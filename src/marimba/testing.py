"""Deterministic synthetic model for integration development, not real inference.

Encoder answers follow fixed signal rules (frame level and pitch) so a synthetic fixture
can trigger each decision. docs/window-server.md states them under "Fake server". They
are a test double and say nothing about what the model answers on the same audio.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from threading import Event
from time import perf_counter

import numpy as np
import numpy.typing as npt

from ._diarization import TASKS
from .audio import SAMPLE_RATE, AudioArray, Mask
from .capabilities import Capability
from .decision import Decision
from .model import Marimba
from .questions import Question
from .runtime import RuntimeInfo
from .windows import EncodedWindow, WindowSnapshot

FRAME_SAMPLES = 160
SILENCE_DB = -60.0
SPEECH_DB = -30.0
PITCH_FRAME_SAMPLES = 1024
PITCH_RANGE_HZ = (60.0, 400.0)
VOICED_CORRELATION = 0.6
VOICE_SCALE_OCTAVES = 0.5
PART_VOICED_FRAMES = 4
COUNT_BLOCK_SAMPLES = 8000
COUNT_BLOCK_VOICED_FRAMES = 3
COUNT_GROUP_BLOCKS = 2
COUNT_PROBABILITY = 0.7
TRAIL_FRAMES = 25
TRAIL_BODY_FRAMES = 100
TRAIL_BODY_MIN_FRAMES = 10
TRAIL_SCALE_DB = 6.0
_PAIR_SPLIT_SECONDS = {"same_speaker": 4.0, "speaker_change": 6.0}

Floats = npt.NDArray[np.float64]


def frame_power(audio: AudioArray, mask: Mask) -> tuple[Floats, Mask]:
    """Mean-square power of each 10 ms frame, and which frames are fully valid."""
    count = len(audio) // FRAME_SAMPLES
    size = count * FRAME_SAMPLES
    frames = audio[:size].astype(np.float64).reshape(count, FRAME_SAMPLES)
    valid: Mask = np.asarray(mask[:size].reshape(count, FRAME_SAMPLES).all(axis=1), dtype=np.bool_)
    return np.mean(frames**2, axis=1), valid


def frame_activity(power: Floats) -> Floats:
    """Zero at -60 dBFS and below, one at -30 dBFS and above, linear in dB between."""
    level = 10 * np.log10(power + 1e-12)
    return np.clip((level - SILENCE_DB) / (SPEECH_DB - SILENCE_DB), 0.0, 1.0)


def frame_pitch(audio: AudioArray) -> Floats:
    """log2 of the pitch in Hz of each 64 ms frame; NaN where the frame is unvoiced."""
    count = len(audio) // PITCH_FRAME_SAMPLES
    if count == 0:
        return np.empty(0, dtype=np.float64)
    size = count * PITCH_FRAME_SAMPLES
    frames = audio[:size].astype(np.float64).reshape(count, PITCH_FRAME_SAMPLES)
    power = np.mean(frames**2, axis=1)
    spectrum = np.abs(np.fft.rfft(frames, 2 * PITCH_FRAME_SAMPLES, axis=1)) ** 2
    correlation = np.fft.irfft(spectrum, axis=1)
    shortest = int(np.ceil(SAMPLE_RATE / PITCH_RANGE_HZ[1]))
    longest = int(SAMPLE_RATE / PITCH_RANGE_HZ[0])
    # Dividing by the zero-lag value leaves the bias toward short lags, which is what
    # picks the period itself rather than one of its multiples.
    scores = correlation[:, shortest : longest + 1] / np.maximum(correlation[:, :1], 1e-12)
    lag = shortest + np.argmax(scores, axis=1)
    voiced = (frame_activity(power) > 0) & (scores.max(axis=1) >= VOICED_CORRELATION)
    return np.where(voiced, np.log2(SAMPLE_RATE / lag), np.nan)


def voice(audio: AudioArray, minimum_frames: int = 1) -> float | None:
    """Mean log2 pitch over the voiced frames, or None with too few of them."""
    pitch = frame_pitch(audio)
    pitch = pitch[~np.isnan(pitch)]
    return float(pitch.mean()) if len(pitch) >= max(1, minimum_frames) else None


def voice_distance(reference: AudioArray, segment: AudioArray) -> float | None:
    """Pitch distance between two parts in octaves, or None when either has too little voice."""
    left, right = voice(reference, PART_VOICED_FRAMES), voice(segment, PART_VOICED_FRAMES)
    return None if left is None or right is None else abs(left - right)


def speaker_groups(audio: AudioArray) -> int:
    """Number of distinct voices among the half-second blocks of the window."""
    anchors: list[float] = []
    sizes: list[int] = []
    for start in range(0, len(audio) - COUNT_BLOCK_SAMPLES + 1, COUNT_BLOCK_SAMPLES):
        value = voice(audio[start : start + COUNT_BLOCK_SAMPLES], COUNT_BLOCK_VOICED_FRAMES)
        if value is None:
            continue
        for index, anchor in enumerate(anchors):
            if abs(value - anchor) <= VOICE_SCALE_OCTAVES:
                sizes[index] += 1
                break
        else:
            anchors.append(value)
            sizes.append(1)
    return sum(size >= COUNT_GROUP_BLOCKS for size in sizes)


def trailing_drop_db(audio: AudioArray, mask: Mask) -> float:
    """How many dB the last quarter second of sound sits below the sound before it."""
    power, valid = frame_power(audio, mask)
    sound = np.flatnonzero(valid & (frame_activity(power) > 0))
    if len(sound) < TRAIL_FRAMES + TRAIL_BODY_MIN_FRAMES:
        return 0.0
    tail = power[sound[-TRAIL_FRAMES:]]
    body = power[sound[-TRAIL_FRAMES - TRAIL_BODY_FRAMES : -TRAIL_FRAMES]]
    return max(0.0, float(10 * np.log10(body.mean() / tail.mean())))


def _ratio(value: float, scale: float) -> float:
    return value**2 / (value**2 + scale**2)


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
            if cap is not None and (
                cap.status == "unavailable" or (question.task in TASKS and not mask.all())
            ):
                decisions.append(
                    Decision(
                        question,
                        question.task or "",
                        question.kind,
                        None,
                        threshold=question.resolved_threshold,
                        window_start_s=start_s,
                        window_end_s=end_s,
                        status=cap.status,
                    )
                )
                continue
            probabilities: dict[str, float] | None = None
            value: bool | float | str
            signal = self._signal_probability(question, window, mask)
            if isinstance(signal, dict):
                probabilities = signal
                value = max(signal, key=signal.__getitem__)
                probability = signal[value]
            else:
                if signal is None:
                    digest = hashlib.sha256(question.text.encode()).digest()
                    offset = int.from_bytes(digest[:4], "big") / 2**32
                    signal = float(min(1.0, energy * 8 + offset * 0.1))
                probability = value = signal
                if question.kind == "noul":
                    value = probability >= question.resolved_threshold
                elif question.kind == "choice":
                    options = question.options or ()
                    weights = np.arange(1, len(options) + 1, dtype=np.float64) + probability
                    weights /= weights.sum()
                    choices = dict(zip(options, map(float, weights), strict=True))
                    probabilities = choices
                    value = max(choices, key=lambda key: choices[key])
                    probability = choices[value]
            decisions.append(
                Decision(
                    question,
                    question.task or "",
                    question.kind,
                    value,
                    probability,
                    probabilities,
                    probability,
                    question.resolved_threshold,
                    max(start_s, end_s - question.seconds)
                    if question.seconds is not None
                    else start_s,
                    end_s,
                    status="experimental",
                )
            )
        return self._apply_calibrations(decisions)

    @staticmethod
    def _signal_probability(
        question: Question, window: AudioArray, mask: Mask
    ) -> float | dict[str, float] | None:
        """Encoder answers from the signal rules; None for tasks that keep the hash rule."""
        task = question.task
        if task == "voice_activity":
            power, valid = frame_power(window, mask)
            if question.seconds is not None:
                tail_frames = round(question.seconds * SAMPLE_RATE) // FRAME_SAMPLES
                valid[:-tail_frames] = False
            return float(frame_activity(power[valid]).mean()) if valid.any() else 0.0
        if task == "turn_end":
            return _ratio(trailing_drop_db(window, mask), TRAIL_SCALE_DB)
        if task in _PAIR_SPLIT_SECONDS:
            split = round(_PAIR_SPLIT_SECONDS[task] * SAMPLE_RATE)
            distance = voice_distance(window[:split], window[split:])
            if distance is None:
                return 0.0
            change = _ratio(distance, VOICE_SCALE_OCTAVES)
            return change if task == "speaker_change" else 1.0 - change
        if task == "speaker_count" and question.options:
            chosen = ("0", "1", "2", "3+")[min(speaker_groups(window), 3)]
            rest = (1.0 - COUNT_PROBABILITY) / (len(question.options) - 1)
            return {
                option: COUNT_PROBABILITY if option == chosen else rest
                for option in question.options
            }
        return None
