"""Exercise Pipecat's public turn analyzer API with synthetic PCM on CPU."""

import asyncio

import numpy as np

from marimba import Marimba
from marimba.integrations.pipecat import MarimbaTurnAnalyzer

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320


async def main():
    model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
    analyzer = MarimbaTurnAnalyzer(model=model)
    analyzer.set_sample_rate(SAMPLE_RATE)
    print("Synthetic tones are not speech; probabilities are a wiring smoke test, not an eval.")
    try:
        for frequency in (180, 300):
            analyzer.clear()
            time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
            tone = 0.2 * np.sin(2 * np.pi * frequency * time) * np.sin(np.pi * time) ** 2
            audio = np.concatenate([tone, np.zeros(SAMPLE_RATE // 2)])
            pcm = (audio * 32767).astype("<i2")
            for start in range(0, len(pcm), CHUNK_SAMPLES):
                analyzer.append_audio(
                    pcm[start : start + CHUNK_SAMPLES].tobytes(),
                    is_speech=start < SAMPLE_RATE,
                )
            state, metrics = await analyzer.analyze_end_of_turn()
            if metrics is None:
                raise RuntimeError("Expected turn metrics for nonempty synthetic audio")
            print(
                f"tone_hz={frequency} state={state.name} "
                f"probability={metrics.probability:.6f} "
                f"latency_ms={metrics.e2e_processing_time_ms:.2f}"
            )
    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
