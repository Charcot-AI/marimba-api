# OpenAI-compatible server

`marimba serve` exposes a minimal OpenAI-compatible HTTP API so routers and OpenAI SDKs
can send one audio clip plus typed questions and receive decisions as JSON assistant
content. It wraps the public package API (`Marimba.from_pretrained`, `Question`, and
`model(audio, questions)`); inference, lanes, and Term decoding (greedy plus four beam
hypotheses) are unchanged.

```bash
uv sync --extra torch --extra server
uv run marimba serve --device cpu --backend torch --port 8000   # real weights
uv run marimba serve --fake --port 8000                         # synthetic decisions
```

Set `MARIMBA_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`.
Other flags: `--host`, `--model-id`, `--prompt-price`, `--completion-price`
(USD strings per token), `--datacenters` (ISO country codes), `--max-concurrency`, and
`--ready` (advertise `is_ready: true`; the default is `false`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Decisions for one clip, streaming or not |
| `GET` | `/v1/models` | One OpenRouter provider model document (`schema_version` 2.4) |
| `GET` | `/ping`, `/health` | Liveness for load balancers |

## Request

The last `user` message must contain exactly one `input_audio` part and one or more
`text` parts. System and assistant messages are ignored.

```json
{
  "model": "charcot-ai/marimba",
  "stream": false,
  "messages": [{
    "role": "user",
    "content": [
      {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}},
      {"type": "text", "text": "language? en, pt\nturn ended?\nterm: marimba"}
    ]
  }]
}
```

- `format`: `wav`, `flac`, `ogg`, `mp3`, or `pcm16` (raw little-endian 16-bit mono
  at 16 kHz). Other containers return `400`. `data` is standard base64; a `data:` URL
  prefix is accepted. Audio is downmixed and resampled to 16 kHz.
- Limits: 30 seconds and 16 MiB of audio (`413` above either), one to eight questions.
  Request bodies larger than the base64 audio limit plus 1 MiB are refused with `413`
  before they are buffered.
- Decisions describe the **last eight seconds** of the clip, the validated window.
  Clips shorter than eight seconds are left-padded and masked; readout tasks
  (speaker questions) return `null` values until a full window is available.
- `n` must be 1. `temperature`, `max_tokens`, and other sampling parameters are
  accepted and ignored: decisions are deterministic for a given clip and revision.

### Question convention

The text part is either JSON or one shorthand question per line.

| Line | Task | Kind |
|---|---|---|
| `language?` or `language? en, pt, es` | `language` | `choice` over the listed codes, or all 30 supported codes |
| `turn ended?` / `turn end?` | `turn_end` | `score` |
| `term: marimba` | `term` | `noul` (asks `Did someone say "marimba"?`) |
| `speech?` / `voice activity?` | `voice_activity` | `noul` |
| `same speaker?` | `same_speaker` | `noul` |
| `speaker changed?` / `speaker change?` | `speaker_change` | `noul` |
| `speaker count?` | `speaker_count` | `choice` over `0`, `1`, `2`, `3+` |

Any other line must match one of the package's English routing templates (for example
`Has the speaker finished?` or `Did someone say "hello"?`) and uses that task's default
kind. For full control, send a JSON question object, a JSON list, or
`{"questions": [...]}` with the `Question` fields `kind`, `text`, and optional
`options`, `task`, and `threshold`:

```json
{"kind": "choice", "text": "Which language is spoken?", "options": ["fr", "de"]}
```

Every question is validated before the model runs. Unknown tasks, kinds a task does
not support, malformed JSON questions, `options` on non-choice questions, unsupported
or repeated language options, term questions without a quoted term, and tasks this
deployment cannot answer (for example `pii` without a detector) return `400`. The
error message names the question by position and never repeats its text, options, or
task string.

## Response

The assistant content is compact JSON with one entry per question, in order:

```json
{"decisions": [{"question": "Has the speaker finished?", "task": "turn_end", "kind": "score",
  "value": 0.5258, "probability": 0.5258, "probabilities": null, "threshold": 0.5,
  "status": "validated", "calibrated": false, "window_start_s": 1.24, "window_end_s": 9.24}]}
```

Streaming (`"stream": true`) returns `text/event-stream`: an SSE comment while the
model runs, one chunk carrying the whole content, a final chunk with
`finish_reason: "stop"` and `usage`, then `data: [DONE]`. Errors use the OpenAI shape
`{"error": {"message", "type", "code"}}` with `400`, `401`, `404` (unknown model),
`413`, or `429` (all `--max-concurrency` slots busy; retry). An unexpected failure
returns `500` with a generic message, or a final `data: {"error": ...}` event when
streaming. A client that disconnects mid-stream keeps its concurrency slot until its
inference finishes. If the model cannot load, `marimba serve` exits with a one-line
error and never starts listening.

The server does not log request content. Uvicorn access logs record the method,
path, status, and client address; an unexpected failure logs only the exception type,
without a traceback.

## Token accounting

Marimba has no text-generation tokenizer budget, so `usage` reports deterministic
billing tokens:

- **Audio:** `ceil(12.5 × scored seconds)`, at least 1, where scored seconds is the
  clip duration capped at the eight-second window: one token per 80 ms, at most 100
  per request. Reported in `usage.prompt_tokens_details.audio_tokens`.
- **Text:** words and individual punctuation marks (`\w+|[^\w\s]`) in the text parts
  (`prompt_tokens`) and in the assistant content (`completion_tokens`).

`prompt_tokens = audio tokens + text tokens`, and the same counts are returned in
streaming and non-streaming responses. Audio beyond the scored window is decoded but
not billed.

## Model document

`GET /v1/models` returns `{"data": [document]}` with text and audio input modalities
(prompt pricing per token as USD strings, a 1,024-token text context, base64 audio up
to 30 seconds), a streaming text output modality with completion pricing and a
1,024-token maximum length, quantization, optional datacenters and capacity, and
`is_ready: false` unless `--ready` is passed.
