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

## Pipecat turn analyzer

Install the optional dependency with `pip install 'marimba-audio[pipecat]'` on Python
3.11 or newer. The analyzer loads the default Marimba model on its first
prediction unless you pass a model.
Pipecat's turn analyzer setup supplies the input sample rate; the adapter resamples
to 16 kHz and keeps the latest eight seconds. The model right-aligns shorter audio
and builds its validity mask.

```python
from marimba.integrations.pipecat import MarimbaTurnAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

turn_analyzer = MarimbaTurnAnalyzer(threshold=0.5)
aggregators = LLMContextAggregatorPair(
    LLMContext(),
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)]
        )
    ),
)
```

Use `aggregators.user()` and `aggregators.assistant()` in the pipeline.

## LiveKit turn detector

Install with `pip install 'marimba-audio[torch,livekit]'`. LiveKit Agents 1.6.3 or
newer is required for its streaming turn-detector interface.

```python
from livekit.agents import AgentSession
from livekit.plugins import silero
from marimba.integrations.livekit import MarimbaTurnDetector

session = AgentSession(
    vad=silero.VAD.load(min_silence_duration=0.25),
    turn_handling={"turn_detection": MarimbaTurnDetector(unlikely_threshold=0.5)},
)
```

Install `livekit-plugins-silero` separately for this VAD. Add your STT, LLM, and
TTS providers to the session. Marimba is audio-only and accepts all language
codes without transcript input. The adapter resamples to 16 kHz with soxr HQ,
keeps the latest eight seconds, and starts loading the default model in a
background thread when a stream is opened unless a model is supplied.
Inference runs off the event loop. Failed or empty
predictions return probability 1.0, matching LiveKit's local fail-open behavior;
canceled predictions return 0.0 and stale results are ignored. `flush()` clears
the audio buffer. Set `prediction_timeout` on the detector to configure the
prediction timeout (default: 2 seconds, must be greater than zero). Background
loading may not finish before the first turn; preload and pass a model to
exclude initialization from that turn. No real-time guarantee is claimed.

See [the examples](../examples/livekit/README.md) for a voice agent and an
offline CPU smoke test with synthetic audio.
