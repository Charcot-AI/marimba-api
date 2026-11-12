"""Typed audio decisions using the package's pretrained inference API."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import spaces

# Spaces installs requirements before copying local package sources into its image.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marimba import Marimba, Question  # noqa: E402

QUESTION_TYPES = ["Speech", "Turn ended", "Language", "Term", "Speaker change", "People"]
EXAMPLES = Path(__file__).parent / "examples"


def make_questions(selected: list[str], term: str) -> list[Question]:
    if not selected or any(item not in QUESTION_TYPES for item in selected):
        raise ValueError("Select at least one supported question type.")
    if len(selected) != len(set(selected)):
        raise ValueError("Select each question type only once.")
    if "Term" in selected:
        term = term.strip()
        if not term or len(term) > 100 or any(char in term for char in '"\u201c\u201d\n\r'):
            raise ValueError("Enter a term of 1–100 characters without quotes or line breaks.")
        if not any(char.isalnum() for char in term):
            raise ValueError("The term must contain a letter or number.")
    questions = {
        "Speech": Question.noul("Is someone speaking?", seconds=2.0),
        "Turn ended": Question.score("Has the speaker finished?"),
        "Language": Question.choice(
            "What language is being spoken?", ["English", "Portuguese", "Spanish", "Other"]
        ),
        "Speaker change": Question.noul("Did the speaker change?"),
        "People": Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
    }
    if "Term" in selected:
        questions["Term"] = Question.noul(f'Did someone say "{term}"?')
    return [questions[item] for item in selected]


@lru_cache(maxsize=1)
def load_model() -> Marimba:
    return Marimba.from_pretrained(
        backend="torch", device=os.environ.get("MARIMBA_DEVICE", "cuda"), dtype="float32"
    )


@spaces.GPU(duration=120)
def infer(audio_path: str, questions: list[Question]) -> list[dict[str, Any]]:
    return [decision.to_dict() for decision in load_model()(audio_path, questions)]


def decide(audio_path: str | None, selected: list[str], term: str) -> list[dict[str, Any]]:
    if not audio_path:
        raise gr.Error("Upload or record audio first.")
    try:
        questions = make_questions(selected, term)
    except (TypeError, ValueError) as error:
        raise gr.Error(str(error)) from None
    try:
        return infer(audio_path, questions)
    except Exception:
        raise gr.Error("Inference failed. Check model access and the uploaded audio.") from None


def build_app() -> gr.Blocks:
    with gr.Blocks(delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# Marimba\n"
            "Multilingual typed audio decisions. Upload or record audio, select questions, "
            "and inspect each decision and its probabilities.\n\n"
            "Only the **last eight seconds** are scored; Speech reads the last two seconds "
            "using the package's validated default threshold. Speaker change and people count "
            "require eight valid seconds; shorter clips return no prediction for those tasks. "
            "People counts speakers, not everyone physically present. Language probabilities "
            "are relative to the listed choices; Other groups the remaining supported languages. "
            "Term probabilities are **uncalibrated fractions of five hypotheses** "
            "(greedy + four beams). No real-time guarantee.\n\n"
            "Examples are synthetic speech generated with ElevenLabs. "
            "Do not submit sensitive audio; "
            "uploads are temporarily cached, with no retention or deletion guarantee."
        )
        audio = gr.Audio(
            type="filepath", sources=["upload", "microphone"], label="Upload or record audio"
        )
        questions = gr.CheckboxGroup(
            choices=QUESTION_TYPES, value=["Speech", "Turn ended"], label="Typed questions"
        )
        term = gr.Textbox(value="marimba", max_lines=1, label="Term to detect (used only for Term)")
        output = gr.JSON(label="Decisions and probabilities")
        gr.Button("Decide", variant="primary").click(
            decide, inputs=[audio, questions, term], outputs=output, api_name="decide"
        )
        gr.Examples(
            examples=[
                [str(path), QUESTION_TYPES, "marimba"] for path in sorted(EXAMPLES.glob("*.wav"))
            ],
            inputs=[audio, questions, term],
            cache_examples=False,
            label="Synthetic examples (eight seconds each)",
        )
    return app


if __name__ == "__main__":
    # ZeroGPU optimizes CUDA placement at startup, before a GPU lease is requested.
    load_model()
    build_app().queue(default_concurrency_limit=1, max_size=20).launch(
        share=False,
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        max_file_size="25mb",
    )
