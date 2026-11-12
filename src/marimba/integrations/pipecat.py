"""Pipecat turn analyzer backed by Marimba's turn-end readout."""

from __future__ import annotations

from typing import Any

import numpy as np
import soxr

try:
    from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn
except ModuleNotFoundError as error:
    raise ImportError(
        "Install the optional Pipecat dependency with `pip install 'marimba-audio[pipecat]'`."
    ) from error

from ..audio import SAMPLE_RATE
from ..model import Marimba
from ..questions import Question

_TURN_END_QUESTION = Question.score("Has the speaker finished their turn?", task="turn_end")
_WINDOW_SAMPLES = 8 * SAMPLE_RATE


class MarimbaTurnAnalyzer(BaseSmartTurn):
    """End-of-turn analyzer that asks Marimba whether the speaker has finished.

    Pipecat buffers the user's audio and calls the analyzer at each pause. The
    analyzer resamples the audio to 16 kHz, keeps the last eight seconds, and
    marks the turn complete when the turn-end probability meets ``threshold``.
    """

    def __init__(
        self,
        *,
        model: Marimba | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initialize the analyzer.

        Args:
            model: Marimba model to use. ``None`` loads the default model on the
                first prediction.
            threshold: Turn-end probability at or above which the turn is complete.
            **kwargs: ``sample_rate`` and ``params`` for ``BaseSmartTurn``.

        Raises:
            ValueError: If ``threshold`` is not between 0 and 1.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        super().__init__(**kwargs)
        self._model = model
        self._threshold = threshold

    def _predict_endpoint(self, audio_array: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Score the buffered audio and return Pipecat's prediction dict.

        Args:
            audio_array: Float32 mono audio at the analyzer's sample rate.

        Returns:
            ``prediction`` (1 when complete, else 0) and ``probability``.

        Raises:
            ValueError: If the model returns no turn-end probability.
        """
        audio_array = np.asarray(audio_array, dtype=np.float32)
        input_rate = self.sample_rate or SAMPLE_RATE
        if input_rate != SAMPLE_RATE:
            audio_array = soxr.resample(audio_array, input_rate, SAMPLE_RATE, quality="HQ")

        if len(audio_array) > _WINDOW_SAMPLES:
            audio_array = audio_array[-_WINDOW_SAMPLES:]

        model = self._model
        if model is None:
            model = Marimba.from_pretrained()
            self._model = model

        decisions = model(audio_array, [_TURN_END_QUESTION], sampling_rate=SAMPLE_RATE)
        if not decisions or decisions[0].probability is None:
            raise ValueError("Marimba did not return a turn-end probability")
        probability = float(decisions[0].probability)
        return {
            "prediction": 1 if probability >= self._threshold else 0,
            "probability": probability,
        }
