"""Typed audio decisions using the package's pretrained inference API."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import spaces

# Spaces installs requirements before copying local package sources into its image.
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "demo"):
    sys.path.insert(0, str(directory))
from marimba_demo.cards import available_cards, questions_for  # noqa: E402
from marimba_demo.replay import REPLAY_SILENCE_SECONDS, replay  # noqa: E402

from marimba import Marimba, Question  # noqa: E402
from marimba.audio import decode_audio  # noqa: E402
from space.live import LiveController  # noqa: E402
from space.presentation import card_values, present  # noqa: E402

QUESTION_TYPES = ["Speech", "Turn ended", "Language", "Term", "Speaker change", "People"]
EXAMPLES = Path(__file__).parent / "examples"


def validate_term(term: str) -> str:
    term = term.strip()
    if not term or len(term) > 100 or any(char in term for char in '"\u201c\u201d\n\r'):
        raise ValueError("Enter a term of 1–100 characters without quotes or line breaks.")
    if not any(char.isalnum() for char in term):
        raise ValueError("The term must contain a letter or number.")
    return term


def demo_cards(model: Any, term: str) -> list[Any]:
    term = validate_term(term)
    return [
        replace(card, title=f'"{term}" spoken', text=f'Was "{term}" spoken?')
        if card.task == "term"
        else card
        for card in available_cards(model)
    ]


def demo_questions(model: Any, term: str) -> list[Question]:
    return questions_for(demo_cards(model, term), Question)


def live_enabled() -> bool:
    return os.environ.get("MARIMBA_LIVE") == "1"


def make_questions(selected: list[str], term: str) -> list[Question]:
    if not selected or any(item not in QUESTION_TYPES for item in selected):
        raise ValueError("Select at least one supported question type.")
    if len(selected) != len(set(selected)):
        raise ValueError("Select each question type only once.")
    if "Term" in selected:
        term = validate_term(term)
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


@spaces.GPU(duration=120)
def infer_replay(audio_path: str, term: str) -> list[dict]:
    audio = decode_audio(audio_path)
    if not audio.size or len(audio) > 30 * 16000:
        raise ValueError("Record or upload between 0 and 30 seconds of audio.")
    model = load_model()
    return asyncio.run(replay(model, demo_questions(model, term), audio))


def run_replay(audio_path: str | None, term: str) -> list[dict]:
    if not audio_path:
        raise gr.Error("Upload or record audio first.")
    try:
        validate_term(term)
        return infer_replay(audio_path, term)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    except Exception:
        raise gr.Error("Replay failed. Check model access and the uploaded audio.") from None


def build_app() -> gr.Blocks:
    model = load_model()
    cards = demo_cards(model, "marimba")
    with gr.Blocks(delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# Marimba\n"
            "Multilingual typed audio decisions using the local demo's scheduler and questions. "
            "Record or upload **up to 30 seconds**, then replay every 250 ms hop. "
            "Speech reads the last 2 s; rolling windows are 8 s. Turn ended, Language and Term "
            "run on closed speech runs. "
            "Speaker change requires speech coverage and eight valid seconds. "
            "People counts speakers, not everyone physically present. Language is a closed "
            "set: en / pt / es. Term is **uncalibrated / raw N-best** (greedy + four beams). "
            "No real-time guarantee.\n\n"
            f"Replay appends {REPLAY_SILENCE_SECONDS:g} s of silence so the last utterance "
            "can close, as in a live stream.\n\n"
            "Examples are synthetic speech generated with ElevenLabs. "
            "Do not submit sensitive audio; "
            "uploads are temporarily cached, with no retention or deletion guarantee."
        )
        audio = gr.Audio(
            type="filepath", sources=["microphone", "upload"], label="Record or upload audio"
        )
        term = gr.Textbox(value="marimba", max_lines=1, label="Term to detect (used only for Term)")
        run = gr.Button("Replay audio", variant="primary")
        outputs = []
        for start in range(0, len(cards), 3):
            with gr.Row():
                for card in cards[start : start + 3]:
                    outputs.append(gr.Markdown(card_values([card], {}, [])[0]))
        plot = gr.LinePlot(
            x="window_end_s",
            y="probability",
            color="card",
            y_lim=[0, 1],
            title="Rolling probabilities",
            x_title="Window end (s)",
            y_title="Probability",
        )
        with gr.Accordion("All scheduler events", open=False):
            events = gr.JSON(label="Events")
        run.click(run_replay, [audio, term], events, api_name="replay").success(
            lambda payloads, user_term: present(payloads, demo_cards(model, user_term)),
            [events, term],
            [*outputs, plot],
            api_visibility="private",
        )
        gr.Examples(
            examples=[[str(path), "marimba"] for path in sorted(EXAMPLES.glob("*.wav"))],
            inputs=[audio, term],
            cache_examples=False,
            label="Synthetic examples (eight seconds each)",
        )
        # Keep the one-shot API for existing smoke-test clients.
        questions = gr.CheckboxGroup(choices=QUESTION_TYPES, value=QUESTION_TYPES, visible=False)
        legacy = gr.JSON(visible=False)
        gr.Button(visible=False).click(decide, [audio, questions, term], legacy, api_name="decide")
        if live_enabled():
            controller = LiveController(model, demo_questions)
            state = gr.State(None, time_to_live=60, delete_callback=controller.delete_state)
            gr.Markdown(
                "## Live microphone (L4)\n"
                "One live session at a time; rolling updates coalesce under load."
            )
            microphone = gr.Audio(
                sources=["microphone"], streaming=True, type="numpy", label="Live microphone"
            )

            async def start(session, user_term, request: gr.Request):
                try:
                    return controller.start(session, user_term, request.session_hash)
                except ValueError as error:
                    raise gr.Error(str(error)) from None

            async def stream(chunk, session, user_term, request: gr.Request):
                try:
                    session = await controller.stream(
                        chunk, session, user_term, request.session_hash
                    )
                    return (
                        session,
                        *card_values(
                            demo_cards(model, session.term), session.latest, session.utterances
                        ),
                    )
                except Exception as error:
                    await controller.close(session, aborted=True)
                    message = (
                        str(error)
                        if isinstance(error, ValueError)
                        else "Live inference failed. Stop and restart."
                    )
                    raise gr.Error(message) from None

            async def stop(session):
                if session is None:
                    return (None, *card_values(cards, {}, []))
                try:
                    await controller.close(session)
                    return (
                        None,
                        *card_values(
                            demo_cards(model, session.term), session.latest, session.utterances
                        ),
                    )
                finally:
                    await controller.close(session, aborted=True)

            async def disconnect(request: gr.Request):
                await controller.disconnect(request.session_hash)

            # Streaming connections hold queue slots until recording stops.
            microphone.start_recording(
                start, [state, term], state, queue=False, api_visibility="private"
            )
            microphone.stream(
                stream,
                [microphone, state, term],
                [state, *outputs],
                stream_every=0.25,
                concurrency_limit=None,
                api_visibility="private",
            )
            for event in (microphone.stop_recording, microphone.clear):
                event(stop, state, [state, *outputs], queue=False, api_visibility="private")
            app.unload(disconnect)
    return app


if __name__ == "__main__":
    # ZeroGPU optimizes CUDA placement at startup, before a GPU lease is requested.
    load_model()
    build_app().queue(default_concurrency_limit=1, max_size=20).launch(
        share=False,
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        max_file_size="25mb",
    )
