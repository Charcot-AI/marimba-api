# Marimba

Marimba: typed, calibrated decisions about short and streaming audio windows (choice / yes-no / score), built on a frozen Qwen3-ASR-0.6B encoder.

Pre-release. The inference package, live demo, framework integrations and public benchmark are under construction.

An open-source project by charcot.ai

## Development interface

Install with `uv sync --extra datasets`. Core installation does not install an inference engine.
Weights are not integrated yet: `Marimba.from_pretrained()` raises `NotImplementedError`.
Use the deterministic synthetic model to build integrations without downloads:

```python
import numpy as np
from marimba import Question
from marimba.testing import FakeMarimba

model = FakeMarimba()
questions = [Question.noul("Is someone speaking?")]
decisions = model(np.zeros(16000, dtype=np.float32), questions, sampling_rate=16000)
stream = model.stream(questions)
updates = stream.push(np.zeros(4000, dtype=np.float32))
updates += stream.flush()
```

`Question.score("Has the speaker finished?")` and
`Question.choice("What language?", ["English", "French"])` select other result kinds.
Pass `task=` for custom wording. Quoted terms are exposed as `question.term`.
`Decision.to_dict()` and `StreamUpdate.to_dict()` return JSON-safe dictionaries.

`model.batch(audios, questions, batch_size=16)` accepts decoded audio mappings or files.
`dataset.map(model.map_fn(questions), batched=True)` adds `marimba_0`, `marimba_1`, etc.,
one structured decision column per question. Arrays require their sampling rate;
files and lazy audio decoders carry it. Two-dimensional arrays are samples by channels;
lazy decoder tensors are channels by samples. Audio is mixed to mono and resampled to 16 kHz.

Streaming accepts mono int16 or floating-point samples at 16 kHz, without re-decoding.
Each complete hop emits one update; `flush()` emits a final partial hop once, with no
extra silence added to the clock. Windows retain the latest eight seconds, left-padding
short inputs with a validity mask. Non-16-kHz stream input is rejected: resample before ingest.
`latency_ms` measures synchronized inference for all questions on one assembled window,
excluding capture, decoding, and resampling. Batch currently uses sequential inference;
`batch_size` is reserved for backend batching.

Run `marimba doctor` for optional runtime discovery, or
`marimba bench --fake --questions 4 --hops 100 --warmup 10` for p50/p95 per-hop latency.
Fake results are synthetic, not calibrated predictions. Capability metadata describes
task evaluation, not the presence of weights in this scaffold.
