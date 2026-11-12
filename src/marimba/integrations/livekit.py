from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np
import soxr

try:
    from livekit import rtc
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
    from livekit.agents.voice.turn import TurnDetectionEvent
except ModuleNotFoundError as error:
    raise ImportError(
        "Install the optional LiveKit dependency with `pip install 'marimba-audio[livekit]'`."
    ) from error

from ..audio import SAMPLE_RATE
from ..model import Marimba
from ..questions import Question

_TURN_END_QUESTION = Question.score("Has the speaker finished their turn?", task="turn_end")
_LOGGER = logging.getLogger(__name__)


class MarimbaTurnDetector:
    """Audio-only streaming turn detector for LiveKit Agents."""

    def __init__(self, *, model: Marimba | None = None, unlikely_threshold: float = 0.5) -> None:
        if not 0.0 <= unlikely_threshold <= 1.0:
            raise ValueError("unlikely_threshold must be between 0 and 1")
        self._model = model
        self._threshold = unlikely_threshold
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return "charcot-ai/marimba"

    @property
    def provider(self) -> str:
        return "marimba"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> MarimbaTurnDetectorStream:
        return MarimbaTurnDetectorStream(self)

    async def unlikely_threshold(self, language: str | None) -> float:
        return self._threshold

    async def supports_language(self, language: str | None) -> bool:
        return True

    def _predict(self, audio: np.ndarray[Any, Any], sample_rate: int) -> float:
        if sample_rate != SAMPLE_RATE:
            audio = soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="HQ")
        audio = audio[-8 * SAMPLE_RATE :]
        with self._lock:
            if self._model is None:
                self._model = Marimba.from_pretrained()
            decisions = self._model(audio, [_TURN_END_QUESTION], sampling_rate=SAMPLE_RATE)
        if not decisions or decisions[0].probability is None:
            raise ValueError("Marimba did not return a turn-end probability")
        return float(decisions[0].probability)


class MarimbaTurnDetectorStream:
    def __init__(self, detector: MarimbaTurnDetector) -> None:
        self._detector = detector
        self._audio = np.empty(0, dtype=np.float32)
        self._sample_rate: int | None = None
        self._closed = False
        self._future: asyncio.Future[TurnDetectionEvent] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def model(self) -> str:
        return self._detector.model

    @property
    def provider(self) -> str:
        return self._detector.provider

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return 30.0

    async def unlikely_threshold(self, language: str | None) -> float:
        return await self._detector.unlikely_threshold(language)

    async def backchannel_threshold(self, language: str | None) -> None:
        return None

    async def supports_language(self, language: str | None) -> bool:
        return await self._detector.supports_language(language)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if self._sample_rate is not None and frame.sample_rate != self._sample_rate:
            raise ValueError("Audio sample rate must remain constant until flush")
        self._sample_rate = frame.sample_rate
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(-1, frame.num_channels).mean(axis=1)
        self._audio = np.concatenate((self._audio, audio))[-8 * frame.sample_rate :]

    @staticmethod
    def _event(probability: float, duration: float | None = None) -> TurnDetectionEvent:
        return TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=probability,
            last_speaking_time=time.time(),
            inference_duration=duration,
        )

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        self.cancel_inference()
        future: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
        if self._closed or not self._audio.size:
            future.set_result(self._event(1.0))
            return future
        self._future = future
        task = asyncio.create_task(
            self._run_prediction(future, self._audio.copy(), self._sample_rate or SAMPLE_RATE)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return future

    async def _run_prediction(
        self,
        future: asyncio.Future[TurnDetectionEvent],
        audio: np.ndarray[Any, Any],
        sample_rate: int,
    ) -> None:
        start = time.monotonic()
        try:
            probability = await asyncio.to_thread(self._detector._predict, audio, sample_rate)
        except Exception:
            _LOGGER.exception("Marimba turn prediction failed")
            probability = 1.0
        if not future.done():
            future.set_result(self._event(probability, time.monotonic() - start))

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        if self._future is not None and not self._future.done():
            self._future.set_result(self._event(0.0))
        self._future = None

    def flush(self, reason: str | None = None) -> None:
        self.cancel_inference()
        self._audio = np.empty(0, dtype=np.float32)
        self._sample_rate = None

    def end_input(self) -> None:
        self.flush()
        self._closed = True

    async def aclose(self) -> None:
        self.end_input()
        # A running model call cannot be interrupted safely in its worker thread.
        if self._tasks:
            await asyncio.gather(*self._tasks)
