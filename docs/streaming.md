# Streaming

Decode and resample once before splitting a recording into chunks:

```python
from marimba.audio import decode_audio

samples = decode_audio("speech.wav")
stream = model.stream(questions, hop_seconds=0.25, coalesce=True)
for start in range(0, len(samples), 4000):
    for update in stream.push(samples[start : start + 4000]):
        print(update.to_dict())
for update in stream.flush():
    print(update.to_dict())
```

push emits due audio-clock hops. With coalesce=True, a large chunk scores only
the latest due hop, without discarding retained context. flush emits the remaining
partial hop once; it does not append silence to the clock.

For caller-controlled scheduling, `stream.ingest(chunk)` changes only the ring,
and `stream.decide()` immediately scores a copied latest window. It returns None
before any samples arrive. Concurrent ingest cannot change a scored snapshot's
samples or timestamps. decide can rescore a window and restarts hop accounting.

[Lane scheduling](scheduling.md) covers snapshot, encode, decide_encoded,
decide_snapshot, separate encoder/decoder locks, cadence and latency fields.
Use a separate caller-managed worker when needed; automatic push runs synchronously.
Separate locks do not eliminate CPU/GPU contention.

Decoder contexts over eight seconds are **unvalidated**. The API's 30-second guard
and a context-handling smoke check are not evidence of long-context task quality.
No real-time guarantee is claimed; see [benchmarks](benchmarks.md) for measured Mac
CPU latency.
