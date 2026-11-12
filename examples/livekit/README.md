# LiveKit turn detection

Install with `uv sync --extra torch --extra livekit` from the repository root.

## Offline CPU smoke test

Run `uv run examples/livekit/synthetic.py`. No LiveKit server or provider keys
are needed. Model files are downloaded on first use. If `espeak-ng` or `espeak`
is installed, the script generates a complete synthetic sentence and a cut-off
version. Otherwise it generates tones. It prints turn-end probability and
inference time for each clip. These are smoke-test results, not quality or
latency benchmarks; synthetic tones do not represent conversational speech.

## Voice agent

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for your LiveKit
project. `agent.py` uses LiveKit Inference for STT, LLM, and TTS, and downloads
Silero VAD on first use. Run:

```sh
uv run --with livekit-plugins-silero examples/livekit/agent.py download-files
uv run --with livekit-plugins-silero examples/livekit/agent.py dev
```

The detector runs locally and loads Marimba lazily on the first prediction.
Preload a model and pass it to `MarimbaTurnDetector(model=model)` to exclude
download and initialization from the first turn. The stream's prediction timeout
is 30 seconds; CPU latency depends on hardware and model initialization.
VAD silence must be at least 0.25 seconds in this example.
