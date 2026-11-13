"""Cog runner that answers typed questions about one audio file."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

from cog import BaseRunner, Input, Path

WEIGHTS_CACHE = pathlib.Path("/opt/marimba/hf")
MAX_QUESTIONS = 32

# Weights baked at build time make the runtime independent of Hub access.
if WEIGHTS_CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(WEIGHTS_CACHE))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The image runs from the repository checkout rather than an installed wheel.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from marimba import Marimba, Question  # noqa: E402

_FIELDS = {"text", "kind", "options", "task", "threshold", "seconds"}


def parse_questions(raw: str) -> list[Question]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"questions must be a JSON list: {error}") from None
    if not isinstance(values, list) or not 0 < len(values) <= MAX_QUESTIONS:
        raise ValueError(f"questions must be a JSON list of 1 to {MAX_QUESTIONS} objects")
    questions = []
    for value in values:
        if not isinstance(value, dict) or not {"text", "kind"} <= value.keys():
            raise ValueError("Each question needs at least text and kind")
        unknown = value.keys() - _FIELDS
        if unknown:
            raise ValueError(f"Unknown question fields: {sorted(unknown)}")
        question = dict(value)
        if question.get("options") is not None:
            question["options"] = tuple(question["options"])
        questions.append(Question(**question))
    return questions


class Predictor(BaseRunner):
    def setup(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model = Marimba.from_pretrained(backend="torch")

    def run(
        self,
        audio: Path = Input(
            description="Audio file (WAV, FLAC or OGG). Only the last eight seconds are scored."
        ),
        questions: str = Input(
            description="JSON list of questions with text, kind, and optional options and task.",
            default='[{"text": "Is someone speaking?", "kind": "noul"}, {"text": "Has the speaker finished?", "kind": "score"}, {"text": "What language is being spoken?", "kind": "choice", "options": ["English", "Portuguese", "Spanish"]}, {"text": "Did someone say “marimba”?", "kind": "noul"}, {"text": "How many speakers?", "kind": "choice", "options": ["0", "1", "2", "3+"]}]',  # noqa: E501
        ),
    ) -> dict[str, Any]:
        decisions = self.model(str(audio), parse_questions(questions))
        return {
            "decisions": [decision.to_dict() for decision in decisions],
            "model_revision": getattr(self.model, "revision", None),
        }
