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
| 429 | `rate_limited` | all inference slots are busy (`--max-concurrency`, default 4; see [sizing](hosting.md#sizing-for-streaming-clients)) | retry after a short backoff, or skip this hop |
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

See [hosting.md](hosting.md) for the container image, thread pinning, the optional
batched encoder for `/v1/pairs` and
[how many streaming sessions one process holds](hosting.md#sizing-for-streaming-clients).

## Fake server

`marimba serve --fake` answers this contract without model weights or a GPU; in Python
the same answers come from `marimba.testing.FakeMarimba`. Every answer is computed from
the request audio with the fixed rules below, so the same bytes always get the same
answer and a synthetic fixture can trigger each decision.

The fake is a test double for contract and ordering tests. It does not detect speech,
speakers or turns, and its answers say nothing about what the model answers on the same
audio. Do not tune thresholds or timing constants against it.

### Measurements

Samples are floats in `[-1, 1)`: the PCM16 value divided by 32768.

**Level.** The window is cut into 10 ms frames (160 samples, from the first sample).
A frame's level is `L = 10 * log10(mean(x^2))` dBFS and its activity is
`clamp((L + 60) / 30, 0, 1)`: 0 at -60 dBFS and below (RMS 0.001, about 33 PCM16 steps),
1 at -30 dBFS and above (RMS 0.032, about 1,036 steps). A frame with `L > -60` is a
*sound frame*.

**Pitch.** A part of the window is cut into 64 ms frames (1,024 samples, from the first
sample of the part; a remainder is dropped). For each lag `k` from 40 to 266 samples
(400 Hz down to 60 Hz) the frame's score is
`sum(x[n] * x[n + k] for n in 0..1023-k) / sum(x[n]^2)`. The frame is *voiced* when its
level is above -60 dBFS and its best score is at least 0.6; its pitch is `16000 / k` Hz
for the smallest lag with the best score. The *voice* of a part is the mean of
`log2(pitch)` over its voiced frames, so the distance `d` between two voices is in
octaves. A sound outside 60 to 400 Hz is read as a pitch inside that range.

Over HTTP every sample of a window counts. In Python, frames outside a snapshot's valid
mask are left out of the level rules, and the three speaker questions need a full window.

### Rules

| `task` | Probability |
|---|---|
| `voice_activity` | Mean activity of the frames in the last `seconds` of the window, or of the whole window when `seconds` is omitted. |
| `turn_end` | Take the sound frames in time order and ignore the rest. The tail is the last 25 of them (0.25 s) and the body is up to 100 before the tail. `drop = 10 * log10(mean power of the body / mean power of the tail)`, at least 0, and `p = drop^2 / (drop^2 + 6^2)`. With fewer than 10 body frames `p = 0`. |
| `speaker_change` | Reference is the first 6 s, segment the last 2 s. `p = d^2 / (d^2 + 0.5^2)` for the distance `d` between their voices. |
| `same_speaker` | Reference is the first 4 s, segment the last 4 s. `p = 1 - d^2 / (d^2 + 0.5^2)`. |
| `speaker_count` | The window is cut into 16 half-second blocks. A block with at least 3 voiced frames has a voice. In time order, a block joins the first group whose first block is within 0.5 octaves, otherwise it starts a group. The answer is the number of groups with at least 2 blocks, as `0`, `1`, `2` or `3+`, with probability 0.7; the other options get 0.1 each. |

For `speaker_change` and `same_speaker`, a part needs at least 4 voiced frames; when
either part has fewer, both probabilities are 0. `/v1/pairs` applies the same two rules
to each window of the body.

What follows from the rules:

- **Speech and silence.** Audio whose frames are at -30 dBFS or louder reads 1.0;
  digital silence and noise below -60 dBFS read 0.
- **Two voices.** Voices half an octave apart sit at 0.5 for both pair questions. One
  octave apart gives `speaker_change` 0.8 and `same_speaker` 0.2; the same voice gives 0
  and 1.
- **Change peaks.** On a rolling window `speaker_change` is highest when the voice
  boundary sits on the 6 s / 2 s split, that is on the hop whose window ends 2 s after
  the boundary. When a pause separates the two voices, the highest value sits on a split
  inside the pause or at the start of the second voice, and splits inside the pause can
  tie.
- **Turn end.** An utterance whose last 0.25 s is at least 6 dB below what came before
  reads 0.5 or more, and keeps reading so while it is the last sound in the window. An
  utterance that stops at full level reads 0. Level differences between utterances count
  as well, so keep the voices of a fixture at one level.
- **Timing.** With a 1 s tail and threshold 0.1, `voice_activity` turns yes on the first
  hop whose tail holds 0.1 s of speech-level audio and turns no once less than 0.1 s is
  left, about 0.9 s after the sound stops. These are not the model's delays.

### Fixture recipe

Two voices are two harmonic tones whose pitches differ by an octave or more. This
fixture triggers every decision:

```python
import numpy as np

RATE = 16000


def voice(f0, seconds, level=0.1):
    """A harmonic tone at pitch f0 (Hz) with RMS level 0.1, about -20 dBFS."""
    t = np.arange(round(seconds * RATE)) / RATE
    wave = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    return (level * wave / np.sqrt(np.mean(wave**2))).astype(np.float32)


def silence(seconds):
    return np.zeros(round(seconds * RATE), dtype=np.float32)


def end_turn(utterance):
    """Drop the last 0.3 s by 12 dB so the fake answers turn_end with yes."""
    out = utterance.copy()
    out[-round(0.3 * RATE) :] *= 0.25
    return out


A, B = 100.0, 250.0  # two voices, 1.32 octaves apart
stream = np.concatenate(
    [
        silence(1.0),
        voice(A, 4.0),  # 1.0 to 5.0 s, turn stays open
        voice(B, 4.0),  # 5.0 to 9.0 s, speaker change at 5.0 s
        silence(2.0),
        end_turn(voice(A, 3.0)),  # 11.0 to 14.0 s, ends a turn
        silence(3.0),
    ]
)
pcm = (stream * 32767).astype("<i2").tobytes()
```

Sent as rolling windows every 0.5 s (a stream shorter than 8 s is padded on the left
with zeros), the fake answers:

| Window end | `voice_activity` (1 s) | `turn_end` | `speaker_change` | `speaker_count` | Why |
|---:|---:|---:|---:|---|---|
| 1.0 s | 0.00 | 0.00 | 0.00 | `0` | silence |
| 1.5 s | 0.50 | 0.00 | 0.00 | `0` | half the tail holds voice A |
| 2.0 s | 1.00 | 0.00 | 0.00 | `1` | voice A fills two blocks |
| 6.0 s | 1.00 | 0.00 | 0.64 | `2` | voice B fills two blocks |
| 7.0 s | 1.00 | 0.00 | 0.87 | `2` | change peak: split at 5.0 s |
| 9.0 s | 1.00 | 0.00 | 0.76 | `2` | voice B stops at full level |
| 10.0 s | 0.00 | 0.00 | 0.64 | `2` | 1 s of silence |
| 13.0 s | 1.00 | 0.00 | 0.87 | `2` | change peak: split at 11.0 s |
| 14.5 s | 0.48 | 0.80 | 0.73 | `2` | voice A trailed off |
| 16.5 s | 0.00 | 0.80 | 0.00 | `1` | voice B left the window |

As pairs, with `a = voice(A, 4.0)` and `b = voice(B, 4.0)`: `same_speaker` answers 1.0
for `(a, a)` and 0.125 for `(a, b)`; `speaker_change` answers 0.0 and 0.875.

Synthetic speech also works when the two voices are far apart in pitch. With
espeak-ng, `-v en-us -p 50` (about 100 Hz) against `-v en-us+f3 -p 99` (about 330 Hz)
read as speech (1 s `voice_activity` between 0.5 and 1.0 while speaking, at about
-20 dBFS), gave `same_speaker` 0.96 or more within a voice and 0.09 or less across
voices, and a `speaker_change` peak of 0.92 on the hop 2 s after the change. Speech moves
its pitch and level all the time, so `turn_end` and `speaker_count` on speech are
erratic: use the tone fixture for those.
