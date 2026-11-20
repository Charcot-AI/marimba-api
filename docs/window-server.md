# Stateless window server: wire contract

`marimba serve` exposes two binary endpoints for streaming pipelines that keep their own
state and ask Marimba one decision at a time. The server holds no per-client state: every
request carries all the audio it needs, any replica can answer any request, and requests
are idempotent, so a failed request can be retried safely.

This page is the complete contract. A client can be written from it alone.

| Endpoint | Body | Answers |
|---|---|---|
| `POST /v1/windows` | one 8 s window | a list of window questions about that window |
| `POST /v1/pairs` | 1 to 32 assembled 8 s windows | one pair question, once per window |
| `GET /health` | none | liveness, never authenticated |

The OpenAI-compatible endpoint in [openai-server.md](openai-server.md) is unchanged and is
served by the same process.

## Transport

- HTTP/1.1 over TCP. The default listener is plain HTTP on port 8000. Started with
  `--tls-certfile` and `--tls-keyfile`, the same port serves HTTPS only; see
  [hosting.md](hosting.md#tls) for certificates and client pinning.
- Connections are persistent. Send requests one after another on the same connection (no
  pipelining) and do not send `Connection: close`. The server closes a connection that has
  been idle for 75 seconds (`--keep-alive-seconds`), so a client must reconnect and resend
  when a reused connection turns out to be closed.
- The request body is raw bytes. Nothing is base64 encoded and there is no multipart
  framing. Send `Content-Length`; `Transfer-Encoding: chunked` is also accepted.

## Audio format

One window is exactly **256,000 bytes**: 128,000 samples of signed 16-bit little-endian
PCM, mono, 16,000 Hz, with no header (no WAV container). Sample `i` occupies bytes `2*i`
and `2*i + 1`, low byte first, and covers time `i / 16000` seconds from the window start.
The server converts each sample to float by dividing by 32768.

To produce PCM16 from float samples in `[-1, 1]`, use
`round(clip(x, -1, 1) * 32767)`. Resample to 16 kHz and downmix to mono before sending;
the server never resamples these bodies.

The whole window is treated as valid audio. If the stream has produced less than eight
seconds so far, wait, or pad on the left knowing that the padding is scored as audio.

## Request headers

| Header | Required | Value |
|---|---|---|
| `Content-Type` | yes | `application/octet-stream` (parameters after `;` are ignored) |
| `Marimba-Questions` | yes | JSON array of question objects, ASCII, at most 4096 bytes |
| `Authorization` | only when the server has a token | `Bearer <token>` |
| `Content-Length` | recommended | exact body size in bytes |

### Bearer token

The check is optional and off by default. When the server is started with the environment
variable `MARIMBA_API_KEY` set to a nonempty value, every `POST` on this page must carry
`Authorization: Bearer <token>` where `<token>` equals that value byte for byte (`Bearer`,
one space, the token). Any other value, or a missing header, returns `401`. When the
server has no token, the header is ignored. `GET /health` never requires it.

### Question objects

`Marimba-Questions` is a JSON array. Each element is an object with these fields:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `task` | string | yes | one of the tasks below |
| `kind` | string | no | one of the kinds the task supports; defaults to the first one listed |
| `threshold` | number in `[0, 1]` | no | decision threshold for `noul`; not allowed for `speaker_count` |
| `seconds` | number, `1` or `2` | no | `voice_activity` only: score only the last 1 or 2 seconds |

Unknown fields are rejected. Questions carry no free text.

| `task` | Kinds | Default threshold | What the probability means |
|---|---|---|---|
| `voice_activity` | `noul` | 0.225 | someone is speaking in the window, or in its last `seconds` |
| `turn_end` | `score`, `noul` | 0.5 | the speaker has finished their turn at the end of the window |
| `speaker_change` | `noul`, `score` | 0.5 | the speaker of the last 2 s differs from the speaker of the first 6 s |
| `speaker_count` | `choice` | none | distribution over `"0"`, `"1"`, `"2"`, `"3+"` speakers in the window |
| `same_speaker` | `noul`, `score` | 0.5 | the first 4 s and the last 4 s are the same speaker |

Kinds decide the type of `value` in the response: `noul` is a boolean
(`probability >= threshold`), `score` is the probability itself, and `choice` is the most
probable option.

## `POST /v1/windows`

One window, several questions, one shared encoder pass.

- Body: exactly 256,000 bytes (one window).
- `Marimba-Questions`: 1 to 8 question objects; any task from the table above.
- All questions are answered on the same window. The response lists one decision per
  question, in request order.

For a live stream, send the most recent eight seconds at every hop. For `speaker_change`
the window is read as `[6 s reference | 2 s segment]` and for `same_speaker` as
`[4 s reference | 4 s segment]`, so a caller may also assemble those windows from
non-contiguous audio.

```bash
curl -sS http://127.0.0.1:8000/v1/windows \
  -H 'Content-Type: application/octet-stream' \
  -H 'Marimba-Questions: [{"task":"voice_activity","seconds":1,"threshold":0.1},{"task":"turn_end"},{"task":"speaker_change","kind":"score"},{"task":"speaker_count"}]' \
  --data-binary @window.pcm
```

## `POST /v1/pairs`

Several assembled windows, one pair question. Use it to compare one segment against many
references (or many segments against one reference) in a single request.

- Body: `N` windows concatenated with no separator, `1 <= N <= 32`, so the body is exactly
  `N * 256000` bytes. Window `k` (zero-based) is bytes `k * 256000` to
  `(k + 1) * 256000 - 1`.
- `Marimba-Questions`: exactly one question object whose `task` is `same_speaker` or
  `speaker_change`.
- The response lists one decision per window, in body order.

The caller assembles each window; the server never stores references.

| `task` | Reference | Segment |
|---|---|---|
| `same_speaker` | samples 0 to 63,999 (first 4 s) | samples 64,000 to 127,999 (last 4 s) |
| `speaker_change` | samples 0 to 95,999 (first 6 s) | samples 96,000 to 127,999 (last 2 s) |

Each part should hold speech from a single speaker. When a part has more audio than its
slot, keep the last seconds of a reference and the first seconds of a segment. When it has
less, repeat it until the slot is full rather than padding with silence. The Python helper
`marimba.assemble_pair(task, reference, segment)` applies exactly these rules.

```bash
cat pair_0.pcm pair_1.pcm pair_2.pcm | curl -sS http://127.0.0.1:8000/v1/pairs \
  -H 'Content-Type: application/octet-stream' \
  -H 'Marimba-Questions: [{"task":"same_speaker","kind":"score"}]' \
  --data-binary @-
```

## Success response

Status `200`, `Content-Type: application/json`, UTF-8.

```json
{
  "object": "window.decisions",
  "model": "charcot-ai/marimba",
  "decisions": [
    {
      "task": "voice_activity",
      "kind": "noul",
      "value": true,
      "probability": 0.9731,
      "probabilities": null,
      "threshold": 0.1,
      "status": "validated",
      "calibrated": false,
      "window_start_s": 7.0,
      "window_end_s": 8.0
    },
    {
      "task": "speaker_count",
      "kind": "choice",
      "value": "1",
      "probability": 0.91,
      "probabilities": {"0": 0.02, "1": 0.91, "2": 0.06, "3+": 0.01},
      "threshold": 0.5,
      "status": "validated",
      "calibrated": false,
      "window_start_s": 0.0,
      "window_end_s": 8.0
    }
  ],
  "latency_ms": 11.482
}
```

| Field | Type | Meaning |
|---|---|---|
| `object` | string | `"window.decisions"` for `/v1/windows`, `"pair.decisions"` for `/v1/pairs` |
| `model` | string | served model identifier |
| `decisions` | array | `/v1/windows`: one per question, in request order. `/v1/pairs`: one per window, in body order |
| `latency_ms` | number | server-side inference time for the whole request, excluding network and body parsing |

Each decision:

| Field | Type | Meaning |
|---|---|---|
| `task` | string | the task that was asked |
| `kind` | string | `"noul"`, `"score"` or `"choice"` |
| `value` | boolean, number or string | boolean for `noul`, number in `[0, 1]` for `score`, option string for `choice`; never `null` |
| `probability` | number in `[0, 1]` | probability of the positive answer, or of the chosen option for `choice` |
| `probabilities` | object or `null` | `choice` only: option to probability, summing to 1; `null` otherwise |
| `threshold` | number | threshold applied to `noul`; reported as `0.5` and unused for `choice` |
| `status` | string | `"validated"` or `"experimental"` (see [capabilities.md](capabilities.md)) |
| `calibrated` | boolean | `true` only when the operator installed a calibration for this question |
| `window_start_s`, `window_end_s` | number | scored span in seconds from the start of the submitted window: `0.0` to `8.0`, or `8.0 - seconds` to `8.0` for `voice_activity` with `seconds` |

New fields may be added to these objects; clients must ignore fields they do not know.

## Errors

Errors on these endpoints use one JSON envelope:

```json
{"error": {"message": "Body must be exactly 256000 bytes", "type": "invalid_request_error", "code": "invalid_audio_length"}}
```

`type` is `"invalid_request_error"` for `4xx` and `"server_error"` for `5xx`. Branch on the
status and `code`; `message` is for humans and may change. Checks run in the order of this
table and the first failure is returned.

| Status | `code` | Cause | Client action |
|---|---|---|---|
| 401 | `invalid_api_key` | the server has a token and `Authorization` does not match | fix the token |
| 415 | `unsupported_media_type` | `Content-Type` is not `application/octet-stream` | fix the header |
| 400 | `invalid_questions` | `Marimba-Questions` is missing, longer than 4096 bytes, not a JSON array, has the wrong number of questions, has an unknown field, task, kind, threshold or `seconds`, or names a task this server cannot answer | fix the request |
| 413 | `request_too_large` | the body is larger than one window (`/v1/windows`) or 32 windows (`/v1/pairs`) | split the request |
| 400 | `invalid_audio_length` | the body is empty or not a whole number of 256,000-byte windows | fix the request |
| 429 | `rate_limited` | all inference slots are busy (`--max-concurrency`, default 4) | retry after a short backoff, or skip this hop |
| 500 | `server_error` | inference failed; no detail is returned | retry once, then alert |

After a `413` the server may close the connection. An unknown path returns
`404 {"detail": "Not Found"}` and a wrong method returns
`405 {"detail": "Method Not Allowed"}`; these two come from the framework and do not use
the envelope.

## Health

`GET /health` (alias `GET /ping`) returns `200 {"status": "healthy"}` once the model has
loaded and the listener is up. It needs no headers.

## Reference client

Standard library only, one persistent connection, one reconnect on a dropped connection.

```python
import http.client
import json


class WindowClient:
    def __init__(self, host: str, port: int = 8000, token: str | None = None) -> None:
        self.address = (host, port)
        self.headers = {"Content-Type": "application/octet-stream"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.connection = http.client.HTTPConnection(*self.address, timeout=5)

    def post(self, path: str, questions: list[dict], body: bytes) -> list[dict]:
        headers = {**self.headers, "Marimba-Questions": json.dumps(questions)}
        for attempt in range(2):
            try:
                self.connection.request("POST", path, body=body, headers=headers)
                response = self.connection.getresponse()
                payload = json.loads(response.read())
                break
            except (http.client.HTTPException, OSError):
                self.connection.close()
                self.connection = http.client.HTTPConnection(*self.address, timeout=5)
                if attempt:
                    raise
        if response.status != 200:
            raise RuntimeError(f"{response.status} {payload['error']['code']}")
        return payload["decisions"]


client = WindowClient("127.0.0.1")
window = bytes(256000)  # eight seconds of PCM16 silence
print(client.post("/v1/windows", [{"task": "voice_activity"}, {"task": "turn_end"}], window))
print(client.post("/v1/pairs", [{"task": "same_speaker", "kind": "score"}], window * 3))
```

## Running the server

```bash
pip install "marimba-audio[torch,server]"
MARIMBA_API_KEY=change-me marimba serve --host 0.0.0.0 --port 8000 --threads 1
```

`marimba serve --fake` answers with deterministic synthetic decisions and needs no model
weights, which is enough to develop a client against this contract. See
[hosting.md](hosting.md) for the container image, thread pinning and the optional batched
encoder for `/v1/pairs`.
