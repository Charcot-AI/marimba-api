"""Local-only upload/question/decision Space scaffold."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import gradio as gr

from marimba import Marimba, Question

DEFAULT_QUESTIONS = json.dumps(
    [
        {"kind": "score", "text": "Has the speaker finished?"},
        {"kind": "noul", "text": "Is someone speaking?"},
    ],
    indent=2,
)


def parse_questions(raw: str) -> list[Question]:
    items = json.loads(raw)
    if not isinstance(items, list) or not 1 <= len(items) <= 8:
        raise ValueError("Provide a JSON list of one to eight questions.")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each question must be an object.")
        result.append(Question(**item))
    return result


@lru_cache(maxsize=1)
def load_model() -> Marimba:
    return Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")


def decide(audio_path: str | None, raw: str) -> list[dict[str, Any]]:
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    try:
        questions = parse_questions(raw)
    except (TypeError, ValueError) as error:
        raise gr.Error("Invalid questions: " + str(error)) from None
    try:
        return [decision.to_dict() for decision in load_model()(audio_path, questions)]
    except Exception:
        raise gr.Error("Inference failed. Check model access and the uploaded audio.") from None


def build_app() -> gr.Blocks:
    with gr.Blocks(delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# Marimba — local scaffold\n"
            "Upload audio and ask supported questions. Only the last eight seconds are scored. "
            "Diarization requires eight valid seconds; partial windows return no prediction. "
            "Term scores are uncalibrated. Decoder context above eight seconds is unvalidated. "
            "No real-time or Mac performance claim. Do not submit sensitive audio."
        )
        audio = gr.Audio(type="filepath", sources=["upload"], label="Audio upload")
        questions = gr.Textbox(value=DEFAULT_QUESTIONS, lines=8, label="Questions (JSON)")
        output = gr.JSON(label="Decisions")
        gr.Button("Decide").click(decide, inputs=[audio, questions], outputs=output)
    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(
        share=False, server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    )
