# Marimba

Marimba: typed, calibrated decisions about short and streaming audio windows (choice / yes-no / score), built on a frozen Qwen3-ASR-0.6B encoder.

Pre-release. The inference package, live demo, framework integrations and public benchmark are under construction.

An open-source project by charcot.ai

## Documentation and release scaffold

Start with the [documentation index](docs/index.md), [executed notebooks](notebooks/README.md),
[model card](hf/README.md), or [local-only Space scaffold](space/README.md).
Decoder context above eight seconds is **unvalidated**. No real-time guarantee is
claimed; see [benchmarks](docs/benchmarks.md) for measured Mac CPU latency. Converted
head/readout weights are Apache-2.0; see the model card.

## Development interface

Install with `uv sync --extra datasets`. Core installation does not install an inference engine.
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
Real language inference takes ISO codes or full English names. An `Other` option
collects the softmax mass of all supported languages not explicitly listed:
`Question.choice("What language is being spoken?", ["English", "Portuguese", "Spanish", "Other"], task="language")`.
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
Fake results are synthetic, not calibrated predictions.

## Pretrained torch lanes

Install `marimba-audio[torch]` (or `uv sync --extra torch`) and authenticate with Hugging Face
using `HF_TOKEN` or `hf auth login` if your model repository requires access:

```python
from marimba import Marimba, Question

model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
questions = [
    Question.score("Has the speaker finished?"),
    Question.choice("What language?", ["en", "pt", "es", "fr", "de"]),
    Question.noul('Did someone say "hello"?'),
]
```

The default manifest revision is pinned in `marimba.weights.DEFAULT_MODEL_REVISION`.
An explicit `revision=` can be a commit, tag, or branch; it is resolved once to an immutable
commit before fetching the manifest and SHA256-verified per-lane files. The frozen
Qwen3-ASR-0.6B encoder revision is pinned separately. Only torch is supported for these
lanes; selecting ONNX fails explicitly. Heavy imports and downloads happen only at load time.

Each window runs the audio tower once. Its native L18 frames, quantized to float16 as in
the readout extraction pipeline, feed valid-frame tail pooling over the last two seconds,
a standardized linear readout, and Platt calibration. Language choices use the sum of
native language-name token log probabilities, softmaxed within the supplied options.
Unsupported options and duplicate aliases raise an error before inference.
Without `Other`, probabilities are normalized within that question's choices;
with `Other`, all 30 supported languages are scored before aggregation.

All term questions share greedy decoding plus four beam hypotheses. Unicode normalization
removes accents and punctuation, retains non-Latin letters, and matches exact token sequences.
**Term probabilities are currently uncalibrated raw fractions of five hypotheses**, not
Platt-calibrated probabilities; their capability is validated and states this limitation.
No calibration is fitted at inference time.

See [Calibrate on your domain](docs/calibration.md) to fit and persist a per-question
Platt map from your own binary labels. This does not train model weights.

Voice activity uses a frozen question-conditioned head, validated with Portuguese prompts
and held-out Arabic, Hindi, Korean and Russian prompts; recognized questions use the
canonical prompt; custom wording passes through unchanged. It uses L9 features
interpolated to 100 frames from the same encoder pass as turn-end.
The head uses the physical valid-frame mask, a frozen Whisper question embedding, a small
text tower, grapheme-encoded English `yes`, and sigmoid on the head logit. Its probabilities
are not additionally temperature-calibrated.

Three L12 encoder readouts share that same pass:

| Task | Evidence / limitation |
|---|---|
| same_speaker | AMI AUROC 0.998, VoxConverse 0.996; both 4 s halves single-speaker; on par with ECAPA |
| speaker_change | dominant speaker in last 2 s vs previous 6 s; AMI AUROC 0.966, VoxConverse 0.980 |
| speaker_count | 0/1/2/3+ speakers; accuracy AMI 0.778, VoxConverse 0.859; undercounts 3+ |

Use `Question.noul("Is this the same speaker?")`,
`Question.score("Did the speaker change?")`, or
`Question.choice("How many speakers?", ["0", "1", "2", "3+"])`.
Both binary lanes support noul/score; count requires exactly those four options
in any order. Diarization requires eight valid seconds: partial or masked
windows return `None` for value, probability and probabilities.
Audio-class is unavailable and does not receive fallback predictions.
PII is experimental when a caller supplies a text detector and unavailable otherwise;
the package does not include a detector. See [PII questions](docs/capabilities.md#pii-questions).
Empty real-model audio raises an error; silence with valid samples is supported.

`marimba bench --per-question --audio public_speech.wav --json bench.json` measures
the four fixed benchmark questions (turn_end, voice_activity, language, and term),
each question alone, and the encoder. The live demo also shows same_speaker,
speaker_change, and speaker_count cards. Warmup and model loading are excluded.
Language options are `en`, `pt`, and `es`.

See [lane scheduling and long context](docs/scheduling.md) for immutable snapshots,
reusable encoded windows, independent encoder/decoder locks, decoder cadence,
and contexts up to `MAX_DECODER_SECONDS`. Direct lane APIs report lane-only
decision latency; `model()` and stream updates retain total-window latency.
Real-model tests require authentication/downloads and are excluded by default; run them
explicitly with `uv run pytest -m real_model`.

## Streaming on slow devices

`stream.ingest(chunk) -> None` only updates the ring buffer. Call
`stream.decide() -> StreamUpdate | None` to score the latest window on demand; it returns
`None` until audio arrives. A decision snapshots the audio, valid mask, and timestamps
under a lock, then releases the lock before inference, so another thread can keep ingesting.
The returned timestamps always describe the snapshot, not audio received during inference.
Repeated `decide()` calls can score the same window; each call restarts hop accounting.

`model.stream(questions, coalesce=True)` makes `push()` emit at most one update: the latest
due hop in that chunk. Trailing samples remain buffered for the next hop or `flush()`.
Skipped hops never clear the context. The default `coalesce=False` still emits every hop.
`ingest()` does not queue historical inference; switching back to `push()` resumes hop
accounting from the retained partial hop. These APIs also work with `FakeMarimba`.
For Pipecat, set `UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MarimbaTurnAnalyzer())])`; see [Pipecat turn analyzer](docs/streaming.md#pipecat-turn-analyzer).
