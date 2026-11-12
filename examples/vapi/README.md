# Vapi custom transcriber

Marimba is not STT. This bridge uses Marimba for speech activity, turn completion,
and typed language decisions; a pluggable callable maps 16 kHz float32 mono audio
to text. The default is the public `openai/whisper-tiny` Transformers pipeline on
CPU, without an API key. Initial model downloads require network access. Marimba
weights must also be accessible to your Hugging Face account if gated.

From the repository root:

```sh
uv run --extra torch --with fastapi --with 'uvicorn[standard]' python examples/vapi/server.py
```

Set `TURN_THRESHOLD` to change the default 0.5 turn-end threshold. Forward local
port 8000 with ngrok or another TLS tunnel, then configure your Vapi assistant:

```json
{
  "transcriber": {
    "provider": "custom-transcriber",
    "server": {"url": "wss://YOUR-TUNNEL.example/transcriber"}
  }
}
```

The first text frame must describe raw, linear16, two-channel PCM. Binary frames
are little-endian interleaved samples: only channel 0 (customer) is retained.
Streaming resampling handles 8–48 kHz input; channel 1 is discarded so the agent
does not hear itself. Arbitrary binary frame boundaries are supported.

Speech activity uses a two-second tail every 250 ms. Only a speech-to-silence
transition triggers turn-end scoring on the last eight seconds, not rolling
turn-end scores every hop. A low score keeps the turn open until a later
speech-to-silence transition. Speech resumption continues the same transcript.
Partials support barge-in, but STT runs at most once per second and synchronously
with inference; slow CPU inference delays both partials and finals. Replace
`whisper_stt()` with a callable for another STT, or increase `partial_seconds`
when constructing `Bridge`. STT receives at most the last 30 seconds of a turn.

Each final also prints a JSON line with typed turn-end and language decisions
and elapsed turn latency. These decisions **cannot** be sent through Vapi's
transcriber protocol; consume stdout separately. This minimal example does not
POST decisions. No audio or transcripts are persisted by the server.

## Limits

This is a local, single-session example, not a production service: add access
control, TLS, bounded queues, timeouts, and an appropriate data-retention policy
before exposing it. Concurrent sessions wait for the current call to disconnect.
Disconnect does not force a final. Send silence after speech to allow endpointing.
There is no real-time guarantee; see [CPU benchmarks](../../docs/benchmarks.md).
Decoder context beyond eight seconds is unvalidated; typed decisions here use
only eight seconds, while STT may cover a longer turn. Language choices are
restricted to English, Portuguese, and Spanish, matching the demo.

Run the synthetic protocol tests with `uv run pytest tests/test_vapi.py`.

To simulate a call with a public or synthetic WAV (never a private recording):

```sh
uv run --with websockets python examples/vapi/simulate_call.py /path/to/public-speech.wav
```

The client sends customer speech plus three seconds of silence, with a separate
assistant-channel tone, and prints every response and total client latency.
The server prints decision JSON lines to its own stdout. The client requires
Python 3.11 or newer for its timeout context manager.
