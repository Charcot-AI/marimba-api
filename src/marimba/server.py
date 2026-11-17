"""OpenAI-compatible Chat Completions server for typed audio decisions.

Requires the ``server`` extra: ``pip install "marimba-audio[server]"``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import json
import logging
import math
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .audio import SAMPLE_RATE, AudioArray, decode_audio
from .capabilities import DEFAULT_CAPABILITIES, Capability
from .decision import Decision
from .language import LANGUAGES, resolve_options
from .lexical import normalize
from .model import Marimba
from .questions import Question, UnsupportedQuestion

AUDIO_TOKENS_PER_SECOND = 12.5
SCORED_WINDOW_SECONDS = 8.0
MAX_AUDIO_SECONDS = 30.0
MAX_AUDIO_BYTES = 16 * 1024 * 1024
# Base64 audio at the audio limit plus room for the JSON envelope and question text.
MAX_REQUEST_BYTES = math.ceil(MAX_AUDIO_BYTES / 3) * 4 + 1024 * 1024
MAX_QUESTIONS = 8
CONTEXT_LENGTH = 1024
MAX_OUTPUT_TOKENS = 1024
AUDIO_FORMATS = ("wav", "flac", "ogg", "mp3", "pcm16")
SPEAKER_COUNT_OPTIONS = ("0", "1", "2", "3+")

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SHORTHANDS = {
    "turn ended?": ("turn_end", "Has the speaker finished?"),
    "turn end?": ("turn_end", "Has the speaker finished?"),
    "speech?": ("voice_activity", "Is someone speaking?"),
    "voice activity?": ("voice_activity", "Is someone speaking?"),
    "same speaker?": ("same_speaker", "Is this the same speaker?"),
    "speaker changed?": ("speaker_change", "Has the speaker changed?"),
    "speaker change?": ("speaker_change", "Has the speaker changed?"),
    "speaker count?": ("speaker_count", "How many speakers are there?"),
}
_QUESTION_KEYS = {"kind", "text", "options", "task", "threshold"}

logger = logging.getLogger(__name__)


class RequestError(ValueError):
    def __init__(
        self, message: str, *, status: int = 400, code: str = "invalid_request_error"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def count_text_tokens(text: str) -> int:
    """Billing tokens for text: words and individual punctuation marks."""
    return len(_TOKEN.findall(text))


def count_audio_tokens(seconds: float) -> int:
    """Billing tokens for audio: 12.5 per scored second, at least one."""
    scored = min(max(seconds, 0.0), SCORED_WINDOW_SECONDS)
    return max(1, math.ceil(round(scored * AUDIO_TOKENS_PER_SECOND, 6)))


def _default_question(task: str, text: str) -> Question:
    kind = DEFAULT_CAPABILITIES[task].kinds[0]
    if task == "language":
        return Question.choice(text, list(LANGUAGES), task=task)
    if task == "speaker_count":
        return Question.choice(text, list(SPEAKER_COUNT_OPTIONS), task=task)
    return Question(text, kind, task=task)


def _parse_line(line: str) -> Question:
    lowered = line.lower()
    if lowered.startswith("language?"):
        options = [item.strip() for item in line[len("language?") :].split(",") if item.strip()]
        text = "What language is being spoken?"
        return Question.choice(text, options or list(LANGUAGES), task="language")
    if lowered.startswith("term:"):
        term = line[len("term:") :].strip().strip("\"“”'")
        if not term:
            raise RequestError("term: requires a word or phrase")
        return Question.noul(f'Did someone say "{term}"?', task="term")
    if lowered in _SHORTHANDS:
        task, text = _SHORTHANDS[lowered]
        return _default_question(task, text)
    probe = Question.noul(line)
    assert probe.task is not None
    return _default_question(probe.task, line)


def _parse_object(item: Any) -> Question:
    if not isinstance(item, Mapping) or not {"kind", "text"} <= item.keys():
        raise RequestError("Each JSON question needs at least kind and text")
    extra = set(item) - _QUESTION_KEYS
    if extra:
        raise RequestError(f"Unknown question fields; allowed: {', '.join(sorted(_QUESTION_KEYS))}")
    options = item.get("options")
    if options is not None and (
        not isinstance(options, list) or not all(isinstance(o, str) for o in options)
    ):
        raise RequestError("options must be a list of strings")
    if options is not None and item["kind"] != "choice":
        raise RequestError("options are only allowed for choice questions")
    threshold = item.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise RequestError("threshold must be a number")
    if not isinstance(item["text"], str) or not isinstance(item["kind"], str):
        raise RequestError("kind and text must be strings")
    task = item.get("task")
    if task is not None and not isinstance(task, str):
        raise RequestError("task must be a string")
    return Question(
        item["text"],
        item["kind"],  # type: ignore[arg-type]
        tuple(options) if options is not None else None,
        task,
        float(threshold),
    )


def parse_questions(text: str) -> list[Question]:
    """Parse the text part: a JSON question, a JSON list, or one shorthand per line."""
    source = text.strip()
    if not source:
        raise RequestError("A text part with at least one question is required")
    items: list[Any]
    parse_item: Any
    try:
        if source[0] in "[{":
            try:
                payload = json.loads(source)
            except json.JSONDecodeError as error:
                raise RequestError(f"Invalid JSON question: {error.msg}") from error
            if isinstance(payload, Mapping) and "questions" in payload:
                payload = payload["questions"]
            items = payload if isinstance(payload, list) else [payload]
            parse_item = _parse_object
        else:
            items = [line.strip() for line in source.splitlines() if line.strip()]
            parse_item = _parse_line
    except RequestError:
        raise
    if not items or len(items) > MAX_QUESTIONS:
        raise RequestError(f"Send between one and {MAX_QUESTIONS} questions")
    questions = []
    for index, item in enumerate(items, 1):
        try:
            questions.append(parse_item(item))
        except RequestError as error:
            raise RequestError(f"Question {index}: {error}") from None
        except (UnsupportedQuestion, ValueError) as error:
            raise RequestError(f"Question {index}: {_question_message(error)}") from None
    return questions


def _question_message(error: ValueError) -> str:
    # Question echoes unmatched text; error bodies never repeat request content.
    if str(error).startswith("No routing template"):
        return "no routing template matches the question text; pass task explicitly"
    return str(error)


def validate_questions(questions: list[Question], capabilities: Mapping[str, Capability]) -> None:
    """Reject questions the model would refuse at inference time, before it runs."""
    for index, question in enumerate(questions, 1):
        cap = capabilities.get(question.task or "")
        if cap is None:
            supported = ", ".join(sorted(capabilities))
            raise RequestError(f"Question {index}: unknown task; supported tasks: {supported}")
        if cap.status == "unavailable":
            raise RequestError(f"Question {index}: task {cap.task} is unavailable here")
        if question.kind not in cap.kinds:
            raise RequestError(
                f"Question {index}: task {cap.task} supports kinds {', '.join(cap.kinds)}"
            )
        if cap.task == "term" and not normalize(question.term or ""):
            raise RequestError(f"Question {index}: term questions need a nonempty quoted term")
        if cap.task == "language":
            try:
                resolve_options(question.options or ())
            except ValueError:
                raise RequestError(
                    f"Question {index}: language options must be distinct supported language "
                    "codes or names, or other"
                ) from None


def decode_input_audio(part: Any) -> AudioArray:
    """Decode an OpenAI ``input_audio`` object into 16 kHz mono float32 samples."""
    if not isinstance(part, Mapping):
        raise RequestError("input_audio must be an object with data and format")
    data, audio_format = part.get("data"), str(part.get("format", "")).lower()
    if audio_format not in AUDIO_FORMATS:
        raise RequestError(f"Unsupported audio format; use one of {', '.join(AUDIO_FORMATS)}")
    if not isinstance(data, str) or not data:
        raise RequestError("input_audio.data must be a base64 string")
    if data.startswith("data:"):
        data = data.partition(",")[2]
    if len(data) > math.ceil(MAX_AUDIO_BYTES / 3) * 4:
        raise RequestError("Audio exceeds 16 MiB", status=413, code="audio_too_large")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RequestError("input_audio.data is not valid base64") from error
    try:
        if audio_format == "pcm16":
            if len(raw) % 2:
                raise ValueError("pcm16 audio needs an even number of bytes")
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            samples = decode_audio(pcm, SAMPLE_RATE)
        else:
            samples = decode_audio(raw)
    except (RuntimeError, ValueError) as error:
        raise RequestError(f"Could not decode {audio_format} audio") from error
    if not samples.size:
        raise RequestError("Audio is empty")
    if samples.size / SAMPLE_RATE > MAX_AUDIO_SECONDS:
        raise RequestError(
            f"Audio is longer than {MAX_AUDIO_SECONDS:g} seconds",
            status=413,
            code="audio_too_long",
        )
    return samples


@dataclass(frozen=True)
class ParsedRequest:
    audio: AudioArray
    questions: list[Question]
    text: str
    stream: bool


def parse_request(body: Any, model_id: str) -> ParsedRequest:
    """Validate a Chat Completions body and extract one audio clip plus questions."""
    if not isinstance(body, Mapping):
        raise RequestError("Request body must be a JSON object")
    if body.get("model", model_id) != model_id:
        raise RequestError(
            "The requested model is not served here", status=404, code="model_not_found"
        )
    if not isinstance(body.get("stream", False), bool):
        raise RequestError("stream must be a boolean")
    if body.get("n", 1) != 1:
        raise RequestError("Only n=1 is supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty list")
    users = [m for m in messages if isinstance(m, Mapping) and m.get("role") == "user"]
    if not users:
        raise RequestError("A user message is required")
    content = users[-1].get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(parts, list):
        raise RequestError("User content must be a string or a list of content parts")
    texts: list[str] = []
    clips: list[Any] = []
    for part in parts:
        kind = part.get("type") if isinstance(part, Mapping) else None
        if kind == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "input_audio":
            clips.append(part.get("input_audio"))
        else:
            raise RequestError("Content parts must be text or input_audio")
    if len(clips) != 1:
        raise RequestError("Send exactly one input_audio part")
    text = "\n".join(texts)
    questions = parse_questions(text)
    return ParsedRequest(decode_input_audio(clips[0]), questions, text, bool(body.get("stream")))


def decision_payload(decision: Decision) -> dict[str, Any]:
    return {
        "question": decision.question.text,
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


def usage(audio_seconds: float, text: str, content: str) -> dict[str, Any]:
    audio_tokens = count_audio_tokens(audio_seconds)
    prompt = audio_tokens + count_text_tokens(text)
    completion = count_text_tokens(content)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"audio_tokens": audio_tokens, "cached_tokens": 0},
    }


@dataclass(frozen=True)
class ServerConfig:
    model_id: str = "charcot-ai/marimba"
    name: str = "Marimba"
    hugging_face_id: str | None = "charcot-ai/marimba"
    description: str = (
        "Multilingual typed audio-decision model. Send one input_audio clip and typed "
        "questions; the assistant replies with JSON decisions about the last eight seconds."
    )
    created: int = 1794441600
    quantization: str = "bf16"
    prompt_price_usd: str = "0.00001"
    completion_price_usd: str = "0.000001"
    datacenters: tuple[str, ...] = ()
    is_ready: bool = False
    api_key: str | None = None
    max_concurrency: int = 4
    requests_per_minute: int | None = None


def model_document(config: ServerConfig) -> dict[str, Any]:
    """One OpenRouter provider model document (schema 2.4)."""
    document: dict[str, Any] = {
        "schema_version": "2.4",
        "id": config.model_id,
        "name": config.name,
        "created": config.created,
        "description": config.description,
        "quantization": config.quantization,
        "is_ready": config.is_ready,
        "input_modalities": [
            {
                "type": "text",
                "supported_inputs": {
                    "max_context_length": {"value": CONTEXT_LENGTH, "unit": "token"}
                },
                "pricing": [
                    {"type": "prompt", "unit": "token", "cost_usd": config.prompt_price_usd}
                ],
            },
            {
                "type": "audio",
                "supported_inputs": {
                    "sources": ["base64"],
                    "formats": ["audio/wav", "audio/flac", "audio/ogg", "audio/mpeg", "pcm16"],
                    "max_duration_seconds": MAX_AUDIO_SECONDS,
                    "max_content_size_bytes": MAX_AUDIO_BYTES,
                },
                "pricing": [
                    {"type": "prompt", "unit": "token", "cost_usd": config.prompt_price_usd}
                ],
            },
        ],
        "output_modalities": [
            {
                "type": "text",
                "streaming": True,
                "max_length": {"value": MAX_OUTPUT_TOKENS, "unit": "token"},
                "supported_parameters": {"max_tokens": {}, "stream": {}},
                "pricing": [
                    {
                        "type": "completion",
                        "unit": "token",
                        "cost_usd": config.completion_price_usd,
                    }
                ],
            }
        ],
    }
    if config.hugging_face_id:
        document["hugging_face_id"] = config.hugging_face_id
    if config.datacenters:
        document["datacenters"] = [{"country_code": code} for code in config.datacenters]
    if config.requests_per_minute:
        document["capacity"] = {"value": config.requests_per_minute, "unit": "request/minute"}
    return document


def _error(message: str, status: int, code: str) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": code}}, status_code=status
    )


_INTERNAL_ERROR = {
    "error": {"message": "Internal server error", "type": "server_error", "code": "server_error"}
}


async def read_body(request: Request) -> bytes:
    """Read the request body, refusing anything over ``MAX_REQUEST_BYTES`` before buffering it."""
    too_large = RequestError("Request body is too large", status=413, code="request_too_large")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise too_large
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def _log_failure(stage: str, error: Exception) -> None:
    # Exception messages and tracebacks can carry request content, so log the type only.
    logger.error("Request %s failed: %s", stage, type(error).__name__)


def _internal_error(stage: str, error: Exception) -> JSONResponse:
    _log_failure(stage, error)
    return JSONResponse(_INTERNAL_ERROR, status_code=500)


def create_app(model: Marimba, config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI app serving ``/v1/chat/completions`` and ``/v1/models``."""
    settings = config or ServerConfig()
    slots = threading.BoundedSemaphore(settings.max_concurrency)
    app = FastAPI(title="Marimba OpenAI-compatible server", docs_url=None, redoc_url=None)

    def authorized(request: Request) -> bool:
        if not settings.api_key:
            return True
        header = request.headers.get("authorization", "")
        return hmac.compare_digest(header.encode(), f"Bearer {settings.api_key}".encode())

    def parse(body: Any) -> ParsedRequest:
        parsed = parse_request(body, settings.model_id)
        validate_questions(parsed.questions, model.capabilities)
        return parsed

    def answer(parsed: ParsedRequest) -> tuple[str, dict[str, Any]]:
        decisions = model(parsed.audio, parsed.questions, sampling_rate=SAMPLE_RATE)
        content = json.dumps(
            {"decisions": [decision_payload(d) for d in decisions]}, separators=(",", ":")
        )
        return content, usage(parsed.audio.size / SAMPLE_RATE, parsed.text, content)

    @app.get("/ping")
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/v1/models")
    def models(request: Request) -> Any:
        if not authorized(request):
            return _error("Invalid API key", 401, "invalid_api_key")
        return {"data": [model_document(settings)]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if not authorized(request):
            return _error("Invalid API key", 401, "invalid_api_key")
        try:
            body = json.loads(await read_body(request))
        except RequestError as error:
            return _error(str(error), error.status, error.code)
        except ValueError:
            return _error("Request body must be valid JSON", 400, "invalid_json")
        try:
            parsed = await run_in_threadpool(parse, body)
        except RequestError as error:
            return _error(str(error), error.status, error.code)
        except Exception as error:
            return _internal_error("validation", error)
        if not slots.acquire(blocking=False):
            return _error("Server is at capacity; retry shortly", 429, "rate_limited")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": completion_id, "created": created, "model": settings.model_id}
        if not parsed.stream:
            try:
                content, counts = await run_in_threadpool(answer, parsed)
            except Exception as error:
                return _internal_error("inference", error)
            finally:
                slots.release()
            return JSONResponse(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": counts,
                }
            )

        def answer_then_release(parsed: ParsedRequest) -> tuple[str, dict[str, Any]]:
            try:
                return answer(parsed)
            finally:
                slots.release()

        # Inference runs outside the response stream so a disconnect, which can cancel or
        # abandon the stream, never frees the slot before inference ends.
        pending = asyncio.get_running_loop().run_in_executor(None, answer_then_release, parsed)
        pending.add_done_callback(lambda done: done.cancelled() or done.exception())

        async def events() -> AsyncIterator[str]:
            chunk = {**base, "object": "chat.completion.chunk"}
            # An SSE comment reaches the client before inference finishes.
            yield ": processing\n\n"
            try:
                content, counts = await pending
            except Exception as error:
                _log_failure("inference", error)
                yield f"data: {json.dumps(_INTERNAL_ERROR)}\n\n"
                return
            delta = {"role": "assistant", "content": content}
            first = {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            yield f"data: {json.dumps(first)}\n\n"
            last = {
                **chunk,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": counts,
            }
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def serve(
    model: Marimba,
    config: ServerConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    import uvicorn

    uvicorn.run(create_app(model, config), host=host, port=port, log_level="info")
