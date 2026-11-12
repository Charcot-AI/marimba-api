"""Send a public or synthetic WAV file as a mock Vapi call."""

import argparse
import asyncio
import json
from time import perf_counter

import numpy as np
import soundfile as sf
import soxr


async def call(path, url):
    from websockets.asyncio.client import connect

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    customer = audio[:, 0]
    customer = soxr.resample(customer, rate, 16000) if rate != 16000 else customer
    customer = np.concatenate((customer, np.zeros(48000, dtype=np.float32)))
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
        started = perf_counter()

        async def send():
            for offset in range(0, len(stereo), 4000):
                await websocket.send(stereo[offset : offset + 4000].tobytes())
                await asyncio.sleep(0.25)

        sender = asyncio.create_task(send())
        try:
            async with asyncio.timeout(600):
                async for message in websocket:
                    print(message, flush=True)
                    if json.loads(message).get("transcriptType") == "final":
                        print(f"Client turn latency: {(perf_counter() - started) * 1000:.1f} ms")
                        break
        finally:
            await sender


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Public or synthetic speech WAV")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/transcriber")
    args = parser.parse_args()
    asyncio.run(call(args.wav, args.url))
