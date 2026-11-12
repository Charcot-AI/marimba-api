# Marimba decoder worker

This Runpod Serverless worker answers Language and Term questions on a CUDA GPU.
It loads Marimba once with the Torch backend and calls `decide_snapshot` without
changing the package's inference path. Term detection retains greedy decoding
plus four beam hypotheses. Encoder questions stay on the client.

## Container and environment

Build from the repository root:

```sh
docker build -f .runpod/Dockerfile -t marimba-decoder:local .
docker run --rm --gpus all --env HF_TOKEN marimba-decoder:local
```

`HF_TOKEN` is an optional Hugging Face read token, required only while model
access requires authentication. Pass it at runtime; never bake it into an image
or put its value in Hub metadata. `HF_HUB_ENABLE_HF_TRANSFER=0` is set in the image
because the base image enables an optional transfer package it does not install.
The worker uses the package's default model and pinned revision.

The Hub configuration selects one GPU from the 24 GB or larger `ADA_24` and
`AMPERE_48` pools, a 30 GB container disk, and CUDA 12.8. Model files download at
startup into the Hugging Face cache; startup requires network and model access.
The Hub smoke test uses one second of synthetic silence and a Language question
on an RTX 4090. It checks execution, not language accuracy on silence.

The files follow the [Runpod Hub publishing guide](https://docs.runpod.io/hub/publishing-guide).
Hub publication is a separate step: it requires accessible source and model
artifacts and a GitHub release. Adding these files does not publish a repository,
model, image, release, or Hub listing.

## Request

Send a Runpod job with an `input` object:

```json
{
  "input": {
    "lane": "decoder",
    "sample_rate": 16000,
    "pcm16_b64": "BASE64_PCM16_LE",
    "mask_b64": "BASE64_UINT8_MASK",
    "window_start_s": 0.0,
    "window_end_s": 1.0,
    "questions": [
      {
        "text": "What language is being spoken?",
        "kind": "choice",
        "options": ["English", "Portuguese"],
        "task": "language",
        "threshold": 0.5
      }
    ]
  }
}
```

`pcm16_b64` encodes mono, signed, little-endian PCM16 at 16 kHz. `mask_b64`
encodes one byte per sample: zero for padding, one for valid audio (including
silence). Both arrays must have the same nonzero sample count, at most 30 seconds,
and at least one sample must be valid. Timestamps are finite seconds with
`0 <= window_start_s <= window_end_s`. Audio may include left padding, so the
valid interval can be shorter than the array duration.

Questions are serialized with `dataclasses.asdict(Question(...))`; JSON options
are restored to tuples. Only `language` and `term` tasks are supported. A Term
example is `Question.noul('Did someone say "hello"?')`. Encoder questions and
other tasks are rejected before loading the GPU model. Invalid inputs fail the
Runpod job rather than producing decisions.

## Response

On success, Runpod's response has `status: "COMPLETED"` and an `output` object:

```json
{
  "decisions": [
    {
      "question": {
        "text": "What language is being spoken?",
        "kind": "choice",
        "options": ["English", "Portuguese"],
        "task": "language",
        "threshold": 0.5
      },
      "task": "language",
      "kind": "choice",
      "value": "English",
      "probability": null,
      "probabilities": {"English": 0.8, "Portuguese": 0.2},
      "score": null,
      "threshold": 0.5,
      "window_start_s": 0.0,
      "window_end_s": 1.0,
      "latency_ms": 0.0,
      "status": "ok",
      "calibrated": false
    }
  ],
  "gpu_compute_ms": 0.0
}
```

The values above illustrate the schema, not a prediction for the synthetic test.
Decisions are returned in question order using `Decision.to_dict()`.
`gpu_compute_ms` times the synchronized `decide_snapshot` call, excluding model
loading, payload decoding, and network time.

## Client usage

Install the repository package with its `torch` extra on the client. Set
`RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` in the client's environment. The API key
authenticates calls to Runpod; it is separate from the worker's `HF_TOKEN`.
The following example creates synthetic silence, keeps voice activity on CPU,
and sends only Language and Term questions to the decoder endpoint:

```python
import base64
import json
import os
from dataclasses import asdict
from urllib.request import Request, urlopen

import numpy as np
from marimba import Marimba, Question, WindowSnapshot

local = Marimba.from_pretrained(device="cpu", backend="torch")
snapshot = WindowSnapshot(
    np.zeros(8 * 16000, dtype=np.float32),
    np.ones(8 * 16000, dtype=np.bool_),
    0.0,
    8.0,
)
questions = [
    Question.score("Is someone speaking?"),
    Question.choice("What language is being spoken?", ["English", "Portuguese"]),
    Question.noul('Did someone say "hello"?'),
]
encoder_questions = [q for q in questions if local.lane(q) == "encoder"]
decoder_questions = [q for q in questions if local.lane(q) == "decoder"]
if any(q.task not in ("language", "term") for q in decoder_questions):
    raise ValueError("The remote worker supports Language and Term only")
local_decisions = local.decide_encoded(local.encode(snapshot), encoder_questions)

pcm16 = np.clip(np.rint(snapshot.audio.astype(np.float64) * 32768), -32768, 32767)
payload = {
    "input": {
        "lane": "decoder",
        "sample_rate": 16000,
        "pcm16_b64": base64.b64encode(pcm16.astype("<i2").tobytes()).decode("ascii"),
        "mask_b64": base64.b64encode(snapshot.mask.astype(np.uint8).tobytes()).decode("ascii"),
        "window_start_s": snapshot.window_start_s,
        "window_end_s": snapshot.window_end_s,
        "questions": [asdict(q) for q in decoder_questions],
    }
}
endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
request = Request(
    f"https://api.runpod.ai/v2/{endpoint}/runsync",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
        "Content-Type": "application/json",
        "User-Agent": "marimba-decoder-client/0.1",
    },
    method="POST",
)
with urlopen(request, timeout=300) as response:
    result = json.load(response)
if result.get("status") != "COMPLETED":
    raise RuntimeError(f"Runpod job {result.get('id')} status: {result.get('status')}")
output = result["output"]
print([d.to_dict() for d in local_decisions])
print(output["decisions"], output["gpu_compute_ms"])
```

For an existing stream, serialize its `WindowSnapshot` in the same way; keep
encoder inference local. A synchronous request may return `IN_QUEUE` or
`IN_PROGRESS`. In that case, retain the job ID and use the
[Runpod status endpoint](https://docs.runpod.io/serverless/endpoints/send-requests)
to retrieve that job instead of submitting duplicate work. Never print API keys
or authorization headers.

## Prior validation

CPU and GPU parity matched for Language and Term in prior validation. On an
RTX 4090, measured median round-trip latency was 663 ms for Language and 1639 ms
for Term; worker GPU compute was 115 ms and 1108 ms, respectively. These are
observations from that validation, not latency guarantees or results of the
synthetic Hub smoke test.
