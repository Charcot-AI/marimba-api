"""Send a public or synthetic WAV file as a mock Vapi call."""

import argparse
import asyncio
import json
from time import perf_counter

import numpy as np
import soundfile as sf
import soxr


async def call(paths, url, drain_seconds):
    from websockets.asyncio.client import connect

    chunks, speech_ends = [], []
    total = 0
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio[:, 0]
        mono = soxr.resample(mono, rate, 16000) if rate != 16000 else mono
        nonzero = np.flatnonzero(np.abs(mono) > 1e-4)
        if not len(nonzero):
            raise ValueError("Expected speech audio, not silence")
        mono = mono[: nonzero[-1] + 1]
        speech_ends.append(total + len(mono))
        chunks.extend((mono, np.zeros(48000, dtype=np.float32)))
        total += len(mono) + 48000
    customer = np.concatenate(chunks)
    assistant = 0.1 * np.sin(np.arange(len(customer)) * 2 * np.pi * 440 / 16000)
    stereo = (np.clip(np.column_stack((customer, assistant)), -1, 1) * 32767).astype("<i2")
    async with connect(url, ping_timeout=None) as websocket:
        await websocket.send(
            json.dumps(
                dict(
                    type="start", encoding="linear16", container="raw", sampleRate=16000, channels=2
                )
            )
        )
        sent_ends = []

        async def send():
            for offset in range(0, len(stereo), 4000):
                await asyncio.sleep(0.25)
                await websocket.send(stereo[offset : offset + 4000].tobytes())
                for end in speech_ends:
                    if offset < end <= offset + 4000:
                        sent_ends.append(perf_counter())

        async def receive():
            async for message in websocket:
                print(message, flush=True)
                if json.loads(message).get("transcriptType") == "final":
                    if sent_ends:
                        latency = (perf_counter() - sent_ends[-1]) * 1000
                        print(f"Client endpoint latency: {latency:.1f} ms", flush=True)
                    else:
                        print("Final arrived before the first speech clip ended", flush=True)

        receiver = asyncio.create_task(receive())
        try:
            await send()
            await asyncio.sleep(drain_seconds)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wav", nargs="+", help="Public or synthetic speech files, joined with silence"
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/transcriber")
    parser.add_argument("--drain-seconds", type=float, default=30.0)
    args = parser.parse_args()
    asyncio.run(call(args.wav, args.url, args.drain_seconds))
