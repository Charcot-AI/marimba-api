"""Serve local assets and stream PCM without persisting audio."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .cards import available_cards, questions_for
from .protocol import SAMPLE_RATE, decode_frame

logger = logging.getLogger(__name__)


def create_app(
    model: Any,
    question_class: Any,
    *,
    static: Path,
    local: Path,
    port: int = 8765,
    stt_config: dict[str, Any] | None = None,
    tasks: set[str] | None = None,
) -> FastAPI:
    cards = available_cards(model)
    if tasks is not None:
        cards = [card for card in cards if card.task in tasks]
    questions = questions_for(cards, question_class)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{host}" for host in hosts}
    busy = asyncio.Lock()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        if request.headers.get("host") not in hosts:
            return JSONResponse({"error": "Loopback host required"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"error": "Same origin required"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self' ws: wss:; worker-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/api/config")
    async def config():
        return {
            "cards": [card.to_dict() for card in cards],
            "runtime": asdict(model.runtime),
            "sample_rate": SAMPLE_RATE,
            "hop_seconds": 0.25,
            "window_seconds": 8,
            "stt": (local / "stt_adapter.js").is_file(),
        }

    @app.get("/api/local-config")
    async def local_config():
        return stt_config or {}

    @app.get("/local/stt_adapter.js")
    async def adapter():
        path = local / "stt_adapter.js"
        if not path.is_file():
            return JSONResponse(
                {"error": "No local adapter installed"}, status_code=404
            )
        return FileResponse(path, media_type="text/javascript")

    @app.websocket("/api/stream")
    async def audio(socket: WebSocket):
        if (
            socket.headers.get("host") not in hosts
            or socket.headers.get("origin") not in origins
        ):
            await socket.close(code=1008)
            return
        await socket.accept()
        if busy.locked():
            await socket.send_json(
                {"type": "error", "message": "Another audio session is active."}
            )
            await socket.close(code=1013)
            return
        async with busy:
            readers: list[asyncio.Task] = []
            finished = False
            connected = True
            aborted = False
            changed = asyncio.Event()
            captures: OrderedDict[float, tuple[int, float]] = OrderedDict()
            send_lock = asyncio.Lock()

            async def send(data):
                async with send_lock:
                    await socket.send_json(data)

            try:
                stream = model.stream(
                    questions,
                    window_seconds=8.0,
                    hop_seconds=0.25,
                    sample_rate=SAMPLE_RATE,
                    coalesce=True,
                )

                async def receive():
                    nonlocal finished, connected
                    previous = -1
                    count = 0
                    try:
                        while True:
                            message = await socket.receive()
                            if message["type"] == "websocket.disconnect":
                                connected = False
                                return
                            if "text" in message:
                                if json.loads(message["text"]) != {"type": "stop"}:
                                    raise ValueError("Unknown control message.")
                                return
                            frame = decode_frame(message.get("bytes", b""))
                            if frame.sequence != previous + 1:
                                raise ValueError("Audio sequence must be continuous.")
                            previous = frame.sequence
                            count += 1
                            captures[count * 0.25] = (
                                frame.sequence,
                                frame.captured_at_ms,
                            )
                            if len(captures) > 128:
                                captures.popitem(last=False)
                            stream.ingest(frame.pcm)
                            changed.set()
                    finally:
                        finished = True
                        changed.set()

                async def decide():
                    previous_end = 0.0
                    coalesced = 0
                    while True:
                        await changed.wait()
                        changed.clear()
                        if aborted or not connected:
                            return
                        update = (
                            await asyncio.to_thread(stream.decide) if captures else None
                        )
                        if aborted or not connected:
                            return
                        if update is not None:
                            capture = captures.get(round(update.window_end_s, 6))
                            if capture is None:
                                raise RuntimeError(
                                    "Inference exceeded capture timestamp history."
                                )
                            sequence, timestamp = capture
                            interval = update.window_end_s - previous_end
                            coalesced += max(0, round(interval / 0.25) - 1)
                            previous_end = update.window_end_s
                            await send(
                                {
                                    "type": "update",
                                    "sequence": sequence,
                                    "captured_at_ms": timestamp,
                                    "decision_interval_ms": interval * 1000,
                                    "coalesced_hops": coalesced,
                                    "decisions": [
                                        d.to_dict() for d in update.decisions
                                    ],
                                }
                            )
                        if finished and not changed.is_set():
                            return

                readers = [
                    asyncio.create_task(receive()),
                    asyncio.create_task(decide()),
                ]
                done, _ = await asyncio.wait(
                    readers, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                await asyncio.gather(*readers)
                if connected:
                    await send({"type": "stopped", "decisions": []})
                    await socket.close()
            except WebSocketDisconnect:
                pass
            except (ValueError, TypeError, KeyError):
                aborted = True
                await send(
                    {"type": "error", "message": "Invalid audio protocol message."}
                )
                await socket.close(code=1003)
            except RuntimeError as error:
                aborted = True
                logger.error("Inference failure: %s", type(error).__name__)
                await send(
                    {
                        "type": "error",
                        "message": "Inference failed. Stop and check the model installation.",
                    }
                )
                await socket.close(code=1011)
            finally:
                aborted = True
                finished = True
                changed.set()
                if readers:
                    readers[0].cancel()
                    await asyncio.gather(*readers, return_exceptions=True)

    app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app
