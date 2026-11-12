# Vapi custom transcriber

Marimba is not STT. This reference bridge uses a cheap customer-channel energy
gate for timing, Marimba for turn completion/voice confirmation/language, and a
pluggable callable (16 kHz float32 mono audio to text) for words. The default is
the public `openai/whisper-tiny` Transformers pipeline on CPU. Downloads need
network access; Marimba weights must be accessible to your account if gated.

From the repository root:

```sh
uv run --extra torch --with fastapi --with 'uvicorn[standard]' python examples/vapi/server.py
```

`GET /health` returns `{"available": true}` (200), or false (503) if model loading
failed. Loading failure logs one sanitized error and leaves the server running;
WebSockets then close with 1011. `MARIMBA_MODEL` optionally selects a model repo.

## Vapi configuration

Forward local port 8000 through a TLS tunnel. For each call, generate a fresh,
opaque `UNIQUE_CALL_KEY` and use the **same key** in both URLs of that call's
transient assistant configuration. Do not share keys between concurrent calls.
The key associates endpointing with a live WebSocket; it is not authentication.

```json
{
  "transcriber": {
    "provider": "custom-transcriber",
    "server": {"url": "wss://YOUR-TUNNEL.example/transcriber?session=UNIQUE_CALL_KEY"}
  },
  "startSpeakingPlan": {
    "waitSeconds": 0.25,
    "smartEndpointingPlan": {
      "provider": "custom-endpointing-model",
      "server": {
        "url": "https://YOUR-TUNNEL.example/endpointing/UNIQUE_CALL_KEY",
        "timeoutSeconds": 3
      }
    }
  }
}
```

Vapi otherwise endpoints on partials independently of this bridge. The endpoint
returns a five-second wait while a turn is active/decoding, and 0.05 seconds
briefly after a final. Missing sessions get the conservative five-second wait.
Call state is removed on disconnect. Use a single server process for this
in-memory association; production deployments need explicit call routing/state.

Sources in the Vapi docs checkout: `fern/customization/voice-pipeline-configuration.mdx:359–405`
defines the custom endpointing configuration, transcript-triggered requests and
resetting timeout; `fern/server-url/events.mdx:583–601` defines the event/response.
The linked Gradium guide recommends this handoff at
`https://docs.gradium.ai/integrations/agent-frameworks/vapi:232–246` (downloaded Markdown line
numbers). Finite endpointing timeouts and CPU latency cannot guarantee that Vapi
always waits for a final; verify a real Vapi call on your hardware.

## Timing and buffering

Only channel 0 is retained from binary, interleaved stereo signed little-endian
16-bit PCM. The first text frame describes `type=start`, `encoding=linear16`,
`container=raw`, `channels=2`, and integer `sampleRate` (8–48 kHz). Streaming
resampling produces 16 kHz. Fragment boundaries need not align to samples.

Every 20 ms customer frame updates an adaptive silence noise floor. Three voiced
frames start a turn, with 200 ms pre-roll; offset is the last voiced frame's end.
After 300 ms quiet, TURN and voice activity score the same eight-second window
ending at offset plus 0.3 seconds, matching the synthetic scene convention
(last speech end plus 0.3 seconds). LANGUAGE uses that same window. Windows are
left-padded with silence. This energy gate is heuristic, not a noise-robust VAD.

`TURN_THRESHOLD` defaults to 0.5. A lower score keeps listening; resumption
continues the turn and invalidates a pending score. At `SILENCE_TIMEOUT` (default
2.5 seconds) the bridge finalizes anyway, **even for a semantically unfinished
sentence**. Voice confirmation must pass for either path; rejected noise does
not produce a final. Decision JSON records `reason=turn_end` or
`reason=silence_timeout`. No later audio is added to an already committed final.
Its STT input contains all samples from turn start/pre-roll through the endpoint;
audio beyond it is retained for the next turn. A stale score cannot cut resumed
speech. Short inter-speaker pauses normally remain part of the same turn.

Partials start after roughly 0.5 seconds of energy-gated speech, at most once per
second. One inference/STT job runs off the event loop per call; receiving continues
and partial requests are coalesced while busy. Finals are sent as soon as STT
finishes, before the language job. CPU inference can still delay partials/finals.

`MAX_TURN_SECONDS` defaults to 30: force-score/split at that boundary with reason
`max_turn` (still requiring voice confirmation). A current turn plus an eight-second
ring bounds audio retention; overload closes with 1013 rather than silently
dropping audio. Binary messages above 1 MiB also close with 1013. Decoder context
above eight seconds is unvalidated; see [CPU benchmarks](../../docs/benchmarks.md).

## Safety and limits

This is a reference, not a production service. Add authentication, TLS, admission
control and an appropriate data-retention policy before exposing it. Concurrent
calls have independent state but share model resources. Unknown text messages
after start are ignored and counted. Invalid starts close with 1003; inference
or receiver failures close with 1011. Disconnect cancels session tasks without
forcing a final; an already-running native inference thread finishes in the
background. Silence-only input retains only the ring and emits no transcripts.

Logs contain only counts, typed scores/labels and timings, never audio or
transcripts. `turn_decisions` JSON lines include language (en/pt/es), TURN,
voice confirmation, final reason and endpoint latency from offset plus 0.3 s.
These decisions **cannot** travel through Vapi's transcriber response protocol;
consume logs separately. No server-side audio/transcript persistence or POST sink
is implemented. Whisper-tiny recognition quality is not guaranteed.

## Local simulation

```sh
uv run pytest tests/test_vapi.py
uv run --with websockets python examples/vapi/simulate_call.py /path/to/public-speech.wav
```

Use public or synthetic audio only. Multiple WAVs are joined with three seconds
silence. The client sends all audio with a distinct assistant-channel tone and
reads until 30 seconds after sending finishes (`--drain-seconds`). Its endpoint
latency starts when the last customer speech sample was sent; near-zero trailing
samples are trimmed at amplitude 0.0001. For multiple clips it uses the latest
completed clip, explicitly labeling finals before the first clip ends. This
client metric and the server's estimated endpoint metric have different origins.
