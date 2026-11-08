# Lane scheduling and long context

`model.lane(question)` returns `"encoder"` for head/readout tasks and `"decoder"`
for language and term tasks. Capability status still applies: routing does not
make an unavailable task available. Same-speaker is unavailable.

```python
import numpy as np
from marimba import Question
from marimba.testing import FakeMarimba

model = FakeMarimba(slow_decoder_seconds=0.1)
voice = Question.noul("Is someone speaking?")
term = Question.noul('Did someone say "hello"?')
stream = model.stream(
    [voice, term],
    context_seconds=30,
    decoder_context_seconds=30,
    decoder_hop_seconds=2.0,
)
stream.ingest(np.zeros(16000, dtype=np.float32))
short = stream.snapshot()
long = stream.snapshot(30)
encoded = model.encode(short)
voice_decisions = model.decide_encoded(encoded, [voice])
term_decisions = model.decide_snapshot(long, [term])
```

## Ownership and concurrency

`WindowSnapshot` is frozen and owns read-only copies of `audio` and `mask`,
with `window_start_s` and `window_end_s`. Snapshots do not consume pending hops.
An empty stream returns `None`. The default duration is `window_seconds`;
explicit durations must fit the retained context. Short history is left-padded,
with a valid-sample mask and timestamps covering only observed audio.

`model.encode(snapshot)` accepts exactly eight seconds and returns a frozen
`EncodedWindow`: `snapshot`, a read-only `features` mapping (L9/L18 where
installed, also exposed as `l9`/`l18`), and `encoder_latency_ms`.
Each result owns its features; subsequent calls cannot overwrite them.

`model.decide_encoded(encoded, questions)` supports either lane. Encoder-only
decisions do not acquire the decoder lock. Decoder decisions reuse the encoded
audio prefix, under a separate decoder lock. Long-context
`model.decide_snapshot(snapshot, questions)` accepts decoder questions only,
up to `MAX_DECODER_SECONDS` (30 seconds). It encodes once, releases the encoder
lock, and then acquires the decoder lock. LID and term questions on the same
snapshot share the audio features and a request-local thinker prefix/KV cache.
There is no global lock around an entire decision, and no shared mutable feature
cache. A decoder can serialize other decoders but cannot hold the encoder lock
while generating. Device contention can still affect latency.

Decoder evaluation evidence is limited to eight-second windows; decoder context
above eight seconds is **unvalidated**. Term remains experimental and uncalibrated.
Encoder/head windows remain eight seconds.
Long-context generation allows a proportionally larger token budget, rather
than retaining the short-window 128-token ceiling.

## Cadence

The ring retains `max(window_seconds, context_seconds)`. With
`decoder_hop_seconds=None`, existing `push()` behavior is unchanged. With a
decoder cadence, encoder questions run each hop; decoder questions run on the
first eligible hop, then when at least that many audio-clock seconds have
elapsed. `decoder_context_seconds` defaults to `context_seconds`; specifying it
without a decoder cadence uses the regular hop cadence. Updates contain only
the decisions computed on that hop, possibly none for a decoder-only stream.
Each decision's timestamps describe its own context; update timestamps describe
the short window. Encoder and decoder contexts end at the same captured time.
When both contexts are eight seconds, they share one encoder pass. A long
decoder context and an eight-second head context need separate encoder passes.

`stream.decide(questions=None)` scores a chosen subset immediately on the short
window, independently of the automatic decoder cadence. `ingest()` is ring-only.
`coalesce=True` still scores only the latest due hop, without clearing context.
`flush()` scores any pending partial hop. Ring locks cover copying/scheduling,
never inference.

## Latency fields

- `EncodedWindow.encoder_latency_ms`: device-synchronized encoder compute,
  excluding time queued behind the encoder lock.
- `Decision.latency_ms` from `decide_encoded` or `decide_snapshot`: synchronized
  lane-group compute only, excluding encoding and time waiting for the decoder
  lock. Questions sharing a lane-group forward receive its elapsed time.
- `model()`, `stream.push()`, `stream.decide()`, and `flush()`: decisions retain
  the total synchronized window latency, including encoding and all lanes run
  in that update, but excluding capture/resampling and snapshot assembly.

Synthetic `FakeMarimba` implements the same ownership, routing, length guards,
cadence, and split locks. `slow_decoder_seconds` simulates decoder work only.
