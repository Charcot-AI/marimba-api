"""Offline CPU smoke test using synthetic audio, not a quality benchmark."""

import asyncio
import io
import shutil
import subprocess

import numpy as np
import soundfile as sf
import soxr
from livekit import rtc

from marimba import Marimba
from marimba.integrations.livekit import MarimbaTurnDetector


def clips() -> list[tuple[str, np.ndarray]]:
    synthesizer = shutil.which("espeak-ng") or shutil.which("espeak")
    if synthesizer:
        wav = subprocess.run(
            [synthesizer, "--stdout", "The package has arrived at the station."],
            check=True,
            capture_output=True,
        ).stdout
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32")
        audio = soxr.resample(audio, rate, 48000, quality="HQ")
        return [("synthetic_complete", audio), ("synthetic_cut_off", audio[: len(audio) // 2])]
    times = np.arange(2 * 48000) / 48000
    tone = (0.1 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
    return [("synthetic_tone", tone), ("synthetic_short_tone", tone[:24000])]


async def main() -> None:
    model = await asyncio.to_thread(
        Marimba.from_pretrained, backend="torch", device="cpu", dtype="float32"
    )
    detector = MarimbaTurnDetector(model=model)
    for name, audio in clips():
        stream = detector.stream()
        try:
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            for start in range(0, len(pcm), 960):
                chunk = pcm[start : start + 960]
                stream.push_audio(
                    rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=48000,
                        num_channels=1,
                        samples_per_channel=len(chunk),
                    )
                )
            event = await stream.predict()
            if event.inference_duration is None:
                raise RuntimeError("No inference was performed")
            print(
                f"{name}: probability={event.end_of_turn_probability:.6f} "
                f"inference_ms={event.inference_duration * 1000:.2f}",
                flush=True,
            )
        finally:
            await stream.aclose()


if __name__ == "__main__":
    asyncio.run(main())
