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

Speech activity uses the latest two-second tail after ingesting all complete
250 ms hops available. Queued frames are coalesced before inference, so CPU
inference does not score every stale hop. Very short activity between checks
can be missed. Only a speech-to-silence transition triggers turn-end scoring.
The eight-second scoring window ends at the estimated speech offset: transition
time minus 1.75 seconds. That voice-activity offset lag was measured on synthetic
scenes in the demo; it is an estimate, not a precise word boundary. Language uses
the same offset-aligned window. A ten-second customer-audio ring retains it.
A low score keeps the turn open until a later
speech-to-silence transition. Speech resumption continues the same transcript.
Partials support barge-in, but STT runs at most once per second and synchronously
with inference; slow CPU inference delays both partials and finals. Replace
`whisper_stt()` with a callable for another STT, or increase `partial_seconds`
when constructing `Bridge`. STT receives at most the last 30 seconds of a turn.

Every transition prints its turn-end score and window end time. Each final also
prints typed turn-end and language decisions and server endpoint latency from
the estimated speech offset to emission (including the offset lag and inference).
The offset wall time is estimated from the latest received audio timestamp.
These decisions **cannot** be sent through Vapi's
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

Pass multiple files to join different clips with three seconds of silence.
The client sends all audio, including trailing silence, with a separate
assistant-channel tone. It prints every response until 30 seconds after sending
finishes (`--drain-seconds`), not just until the first final. Client endpoint
latency starts when the last customer speech sample was sent; near-zero trailing
samples are trimmed at amplitude 0.0001. In a multi-clip call it uses the latest
completed clip. A final before the first clip ends is explicitly labeled.
The server prints decision JSON lines to its own stdout. Clip boundaries and
the server's estimated offsets may differ; neither metric starts at call start.
