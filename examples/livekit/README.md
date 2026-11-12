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

The detector runs locally and starts loading Marimba in a background thread
when a stream is opened.
Preload a model and pass it to `MarimbaTurnDetector(model=model)` to exclude
download and initialization from the first turn. The detector's `prediction_timeout`
defaults to 2 seconds and must be greater than zero; CPU latency depends on
hardware and model initialization. Background loading may not finish before
the first turn.
VAD silence must be at least 0.25 seconds in this example.
