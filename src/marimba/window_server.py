"""Stateless binary endpoints for assembled eight-second windows.

The wire contract is documented in ``docs/window-server.md``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .audio import AudioArray
from .capabilities import DEFAULT_CAPABILITIES, Capability
from .decision import Decision
from .model import Marimba
from .questions import Question, UnsupportedQuestion
from .windows import PAIR_LAYOUTS, WINDOW_SAMPLES

WINDOW_BYTES = WINDOW_SAMPLES * 2
MAX_WINDOW_QUESTIONS = 8
MAX_PAIR_WINDOWS = 32
MAX_QUESTIONS_HEADER_BYTES = 4096
QUESTIONS_HEADER = "marimba-questions"
MEDIA_TYPE = "application/octet-stream"
WINDOW_QUESTION_TEXT = {
    "voice_activity": "Is someone speaking?",
    "turn_end": "Has the speaker finished?",
    "speaker_change": "Has the speaker changed?",
    "speaker_count": "How many speakers are there?",
    "same_speaker": "Is this the same speaker?",
}
_QUESTION_KEYS = {"task", "kind", "threshold", "seconds"}

logger = logging.getLogger(__name__)


class WindowRequestError(ValueError):
    def __init__(self, message: str, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class WindowServerSettings:
    model_id: str
    batched_encoder: bool = False
    encoder_batch_size: int = 16


def _invalid(message: str) -> WindowRequestError:
    return WindowRequestError(message, 400, "invalid_questions")


def _parse_question(item: Any, tasks: tuple[str, ...]) -> Question:
    if not isinstance(item, Mapping) or not isinstance(item.get("task"), str):
        raise _invalid("each question must be an object with a string task")
    if set(item) - _QUESTION_KEYS:
        raise _invalid(f"allowed question fields: {', '.join(sorted(_QUESTION_KEYS))}")
    task = item["task"]
    if task not in tasks:
        raise _invalid(f"task must be one of {', '.join(tasks)}")
    kinds = DEFAULT_CAPABILITIES[task].kinds
    kind = item.get("kind", kinds[0])
    if kind not in kinds:
        raise _invalid(f"task {task} supports kinds {', '.join(kinds)}")
    threshold, seconds = item.get("threshold"), item.get("seconds")
    for name, number in (("threshold", threshold), ("seconds", seconds)):
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, (int, float))
        ):
            raise _invalid(f"{name} must be a number")
    if threshold is not None and (kind == "choice" or not 0 <= threshold <= 1):
        raise _invalid("threshold must be between 0 and 1 and is not allowed for choice")
    try:
        return Question(
            WINDOW_QUESTION_TEXT[task],
            kind,
            ("0", "1", "2", "3+") if task == "speaker_count" else None,
            task,
            None if threshold is None else float(threshold),
            None if seconds is None else float(seconds),
        )
    except (UnsupportedQuestion, ValueError) as error:
        raise _invalid(str(error)) from None


def parse_window_questions(
    header: str | None,
    capabilities: Mapping[str, Capability],
    tasks: tuple[str, ...] = tuple(WINDOW_QUESTION_TEXT),
    limit: int = MAX_WINDOW_QUESTIONS,
) -> list[Question]:
    """Parse the ``Marimba-Questions`` header: a JSON array of question objects."""
    if not header:
        raise _invalid("the Marimba-Questions header is required")
    if len(header) > MAX_QUESTIONS_HEADER_BYTES:
        raise _invalid(f"Marimba-Questions is limited to {MAX_QUESTIONS_HEADER_BYTES} bytes")
    try:
        items = json.loads(header)
    except ValueError:
        raise _invalid("Marimba-Questions must be a JSON array") from None
    if not isinstance(items, list) or not 1 <= len(items) <= limit:
        raise _invalid(f"Marimba-Questions must be a JSON array of 1 to {limit} objects")
    questions = []
    for index, item in enumerate(items, 1):
        try:
            question = _parse_question(item, tasks)
        except WindowRequestError as error:
            raise _invalid(f"question {index}: {error}") from None
        if capabilities[question.task or ""].status == "unavailable":
            raise _invalid(f"question {index}: task {question.task} is unavailable here")
        questions.append(question)
    return questions


def decode_windows(body: bytes, limit: int) -> list[AudioArray]:
    """Split a body of concatenated little-endian PCM16 windows into float32 windows."""
    count, rest = divmod(len(body), WINDOW_BYTES)
    if rest or not 1 <= count <= limit:
        expected = (
            f"exactly {WINDOW_BYTES} bytes"
            if limit == 1
            else f"1 to {limit} windows of {WINDOW_BYTES} bytes each"
        )
        raise WindowRequestError(f"Body must be {expected}", 400, "invalid_audio_length")
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    return list(samples.reshape(count, WINDOW_SAMPLES))


def window_decision_payload(decision: Decision) -> dict[str, Any]:
    return {
        "task": decision.task,
        "kind": decision.kind,
        "value": decision.value,
        "probability": decision.probability,
        "probabilities": decision.probabilities,
        "threshold": decision.threshold,
        "status": decision.status,
        "calibrated": decision.calibrated,
        "window_start_s": round(decision.window_start_s, 3),
        "window_end_s": round(decision.window_end_s, 3),
    }


def _error(message: str, status: int, code: str) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": code}}, status_code=status
    )


async def _read(request: Request, limit: int) -> bytes:
    too_large = WindowRequestError("Request body is too large", 413, "request_too_large")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def add_window_routes(
    app: FastAPI,
    model: Marimba,
    settings: WindowServerSettings,
    slots: threading.BoundedSemaphore,
    authorized: Callable[[Request], bool],
) -> None:
    """Register ``POST /v1/windows`` and ``POST /v1/pairs`` on an existing app."""

    async def handle(request: Request, pairs: bool) -> JSONResponse:
        if not authorized(request):
            return _error("Invalid API key", 401, "invalid_api_key")
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media_type != MEDIA_TYPE:
            return _error(f"Content-Type must be {MEDIA_TYPE}", 415, "unsupported_media_type")
        limit = MAX_PAIR_WINDOWS if pairs else 1
        try:
            questions = parse_window_questions(
                request.headers.get(QUESTIONS_HEADER),
                model.capabilities,
                tuple(PAIR_LAYOUTS) if pairs else tuple(WINDOW_QUESTION_TEXT),
                1 if pairs else MAX_WINDOW_QUESTIONS,
            )
            windows = decode_windows(await _read(request, limit * WINDOW_BYTES), limit)
        except WindowRequestError as error:
            return _error(str(error), error.status, error.code)
        if not slots.acquire(blocking=False):
            return _error("Server is at capacity; retry shortly", 429, "rate_limited")

        def infer() -> tuple[list[list[Decision]], float]:
            started = time.perf_counter()
            rows = model.decide_windows(
                windows,
                questions,
                batched=pairs and settings.batched_encoder,
                batch_size=settings.encoder_batch_size,
            )
            return rows, (time.perf_counter() - started) * 1000

        try:
            rows, latency = await run_in_threadpool(infer)
        except Exception as error:
            # Exception messages can carry request content, so log the type only.
            logger.error("Window inference failed: %s", type(error).__name__)
            return _error("Internal server error", 500, "server_error")
        finally:
            slots.release()
        decisions = [row[0] for row in rows] if pairs else rows[0]
        return JSONResponse(
            {
                "object": "pair.decisions" if pairs else "window.decisions",
                "model": settings.model_id,
                "decisions": [window_decision_payload(decision) for decision in decisions],
                "latency_ms": round(latency, 3),
            }
        )

    @app.post("/v1/windows")
    async def windows(request: Request) -> JSONResponse:
        return await handle(request, False)

    @app.post("/v1/pairs")
    async def pairs(request: Request) -> JSONResponse:
        return await handle(request, True)
