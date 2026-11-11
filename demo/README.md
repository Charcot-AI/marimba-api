# Marimba live instruments

A loopback-only app for microphone or file audio, with parallel decision cards.
The browser sends mono PCM16 at 16 kHz; the server calls the inference package's
`model.stream(questions, window_seconds=8, context_seconds=review_window,
hop_seconds=0.25, sample_rate=16000, coalesce=True)`. Cheap `ingest()` retains every
packet; independent workers score live snapshots and buffered reviews.
It does not save audio or transcripts. A user-installed transcript adapter can
send audio to another service; check that service's retention policy separately.

## Run on an Apple Silicon Mac

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22 LTS and npm.
Use Chrome or Safari with microphone access allowed for localhost. The inference
package and compatible model weights must be available; this app never substitutes
fabricated model outputs for missing weights.

Chrome is recommended for the recording. If a browser rejects a 16 kHz
AudioContext, the worklet converts its native rate to 16 kHz; stereo files are
downmixed before PCM encoding. Hardware/browser behavior still needs checking on
the recording Mac.

From the repository root, install once:

```sh
uv sync --project demo --python 3.12
uv pip install --python demo/.venv/bin/python -e '.[torch]' -e demo
```

Then one command builds the lightweight frontend on first launch, loads the model,
starts Python on `127.0.0.1:8765`, and opens the page:

```sh
uv run --project demo --no-sync marimba-demo
```

Installing both editable projects from this repository root uses the in-repo
inference engine and preserves the demo's dependency constraints. `--no-sync`
keeps that combined installation intact. The frontend uses
Vite and TypeScript, with no UI framework. All fonts and brand assets are local;
there is no CDN, analytics, audio upload, or remote model API in the default app.
Model download may require access credentials until weights are released.

For reproducibility, pass `--revision <model-commit>` and record the runtime shown
at the bottom of the page. `--device cpu` selects CPU; `--device auto --backend auto`
lets the package select a supported accelerator. Apple Silicon acceleration is
not guaranteed: check `marimba doctor` and the displayed device. Never describe a
CPU fallback as an Apple GPU result. `--no-open` suppresses opening the browser;
`--port 8766` uses another loopback port.

Click **Start microphone** or **Open audio**. File audio is played audibly and paced
in real time; it is decoded locally and sent through the same worklet and socket.
Use headphones to avoid recording playback back into the microphone. Stop retains
the last decisions, which dim when stale (2.5 s or twice the review pause). Restart clears
the previous session, its trail, and timing samples. Inputs are not auto-reconnected.

## Live signals and buffered reviews

The **Live signals** grid contains encoder-lane cards only: turn completion,
speech present, speaker change, and speaker count. Each encoder
tick scores the latest eight-second window with `encode()` and `decide_encoded()`.
Decoder work runs independently through `decide_snapshot()`; it must not hold the
encoder/head lock. One review batch runs at a time, coalescing to the latest audio
instead of building a backlog. Language and term questions due together share a
snapshot and the decoder's audio prefix where the package supports it.

The separate **Buffered reviews** section retains older results while computing,
shows each card's observed update interval, result age, exact audio span and its
own model timing. Its p95 is separate from the live target. A minimum pause after
each card completes is configurable; this is not a promise of that update rate:

```sh
uv run --project demo --no-sync marimba-demo \
  --review-window 8 --review-intervals language=2,term=4
```

The default completion pauses are 2 s for language and 4 s for term. These are
conservative: measured standalone p95 on one public speech clip was 2,800/7,155 ms
on a four-thread Linux CPU and 117/1,315 ms on an NVIDIA L4, respectively (fp32,
5 warmups, 40 measured hops, encoder included). Those are model-only measurements,
not Mac or browser capture-to-render results. Longer-context latency must be
measured separately; the eight-second timings do not predict a 30-second review.

`--review-window` accepts 8–30 seconds, subject to the inference package's context
limit. Reviews wait until that much audio has actually accumulated. The eight-second
context remains the validated workload. Larger contexts are opt-in experiments,
visibly **UNVALIDATED WINDOW**; their cards/results are not labelled validated and
are outside the live grid. Successful execution is not an accuracy validation.
Do not claim that a longer buffer improves language or term detection without a
matched public-validation evaluation.

A controlled check used nine 30-second same-language sequences (English, Portuguese
and Spanish) built from the [FLEURS validation split](https://huggingface.co/datasets/google/fleurs)
(CC BY 4.0, dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`,
converted-Parquet revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`).
Language was correct on 9/9 sequences at each of 8, 16 and 30 seconds. All nine
early terms scored 0 outside the 8/16-second controls and 1 inside the 30-second
context. At the raw-support threshold of 0.5, present-term hits were 8/9, 8/9 and
18/18, respectively; no absent term crossed it. These paired checks are too small
for release validation or calibration. Speaker identity is unverified, possibly
mixed; the sequences were not used for speaker metrics.

The package must expose `snapshot`, `encode`, `decide_encoded`, `decide_snapshot`
and `lane` for this split scheduling. Disconnect/abort discards further results;
the current model call may need to finish before another session acquires the model.
Graceful Stop may retain a final in-flight result. It ends capture immediately,
then keeps the controls disabled while current work
drains (bounded to 30–90 s). Transcription flushes independently. A forced disconnect
does not cancel an in-flight model call; wait or restart the server if it remains busy.
The card's **RECEIVED** age is time since delivery; **AUDIO AGE** starts at the
estimated end of its captured window, including compute and transport delay.

## Cards and visual language

The checkpoint's `model.capabilities` is authoritative. Unavailable tasks are
omitted, not drawn with invented answers. Experimental tasks are explicitly marked.
Uncalibrated term support is labelled **UNCALIBRATED / RAW N-BEST**, not presented
as a calibrated probability. The current language card is a closed set
of the ISO codes `en`, `pt`, `es`; it does not claim an open-set "Other" detector.
The demo requests six released tasks, numbered 01–06 without gaps: turn_end,
voice_activity, speaker_change, speaker_count, language, and term.
Capability gating can reduce this set further; unavailable tasks are not requested.
Four live signals and two buffered reviews reflow into equal-width groups.

| Task | Instrument |
|---|---|
| Turn ended | 0–1 ruler, pointer, trailing positions from the last five seconds |
| Speech | P(yes) needle and a yes/no reading |
| Speaker changed | 0–1 ruler, pointer, trailing positions from the last five seconds |
| People | Four-option probability bars for 0/1/2/3+ |
| Language | One thin horizontal probability bar per option |
| Marimba spoken | Support needle; explicitly labelled when uncalibrated |

Position on a shared scale and aligned bar lengths make small comparisons easier
than interpreting colored areas or angles. See Cleveland and McGill,
[Graphical Perception (1984)](https://doi.org/10.1080/01621459.1984.10478080).
This motivates the linear rulers and VU-style bars, rather than decorative dials.
Numbers and labels remain visible; nothing depends on color alone. A binary/score
threshold crossing briefly accents one hairline without a color flash. Motion
respects the system reduced-motion preference. The UI follows the Charcot kit:
IBM Plex, thin geometry, a grid, black/white or warm white, no bold, shadows or gradients.

## Timing: measure on your own machine

These measurements have different boundaries:

- **Per-card model ms:** server-observed wall time around the relevant encoder and
  lane call, including the package's accelerator synchronization, but excluding
  capture, transport and paint. Cards in one batch may share the number. The
  package's additive APIs separately report lane-only `Decision.latency_ms`; the
  demo's `window_latency_ms` includes the relevant encoder, avoiding a head-only
  timing being presented as the whole model duration.
- **Live capture → render p95:** nearest-rank p95 over the last 60 seconds of live
  updates only. It includes the 250 ms packetization interval, socket round trip,
  inference and next animation frame. Capture start is estimated from the worklet
  audio timestamp minus 250 ms; microphone hardware/OS buffering is not directly measured.
  Worklet delivery is mapped to the AudioContext timeline, so main-thread queueing
  is included. This is an app-observed estimate, not a hardware-loopback acoustic measurement.
- **Review window end → render p95:** starts at the estimated end of the captured
  packet (capture-start timestamp plus 250 ms), for buffered review updates only.
  It excludes packetization and collecting the preceding 8–30 s context; the
  exact audio span and context length are shown separately. It is not subject to
  the live ≤300 ms target.

The ≤300 ms end-to-end target is a target, not a guarantee. A missed target is
labelled explicitly. Startup/model download is excluded; allow several warmup hops.
One live worker and one review worker run independently. Audio ingest continues,
so slow inference never repeatedly resets the context or concatenates separated
audio as if continuous. The footer reports the actual spacing of scored windows
and counts coalesced decision ticks, not dropped input. There is no promise of a
decision at every 250 ms hop on slower hardware. If audio transport itself backs
up, the app stops explicitly instead of silently losing audio. A missing or
duplicate sequence closes the session. Socket queues and timestamp history are bounded.

For a lower-load CPU subset without lexical decoding, start with
`marimba-demo --device cpu --tasks turn_end,voice_activity` (using the
same `uv run --project demo --no-sync` prefix). Unsupported tasks are still
omitted. This changes the workload; do not compare its timing to a full-grid run
without listing the different questions.
Language and term both use the slower decoder; removing term alone does not make
language cheap. Choose thread counts/backends from measured results, not assumptions;
do not substitute quantized or alternate-runtime variants without parity validation.

For the package's isolated model-hop benchmark, use its CLI in this environment:

```sh
uv run --project demo --no-sync marimba doctor
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --project demo --no-sync marimba bench --questions 2 --hops 40 --warmup 5
```

The package benchmark has its own templates and synthetic silence input: this
two-question example is not necessarily the demo's exact question/options set.
It avoids the expensive lexical lane in the current package. Inspect its templates
before comparing workloads; repeating a template does not create a new capability.
Use `marimba bench --help` for supported device/backend/model/revision switches in
your installed package. Record the model revision, question list, thread count,
runtime, audio source, number of warmup/measured hops and whether samples were
coalesced. Use the demo overlay for capture-to-render latency: the bench command
does not include browser capture or paint. Mac timings must be measured on the
Mac; Linux CPU and NVIDIA GPU timings cannot be relabelled as Mac results.

## Optional transcription

No adapter is shipped or enabled by default, and the transcript panel stays hidden.
The interface is `frontend/src/stt.ts`: `start(onTranscript, onError)`, `push(pcm,
capturedAtEpochMs)`, `stop()`. `Transcript` events have stable IDs; final events can
replace partial IDs. Text is inserted as text nodes, never HTML.

To use your own adapter, place one file exporting `createAdapter(config)` at
`demo/local/stt_adapter.ts`. The launcher compiles it to `local/stt_adapter.js`
outside the public frontend bundle. `local/` is gitignored and is never packaged.
Set `MARIMBA_STT_CONFIG` to a local JSON object with the adapter's configuration;
the browser fetches it from a no-store, same-origin loopback endpoint. No config
is embedded in the frontend build. Never put credentials in a source file or
commit an adapter implementing a non-public service protocol. Do not expose the
loopback server through a public tunnel; local processes still have access to it.

## Development and checks

```sh
uv sync --project demo --group dev --python 3.12
uv run --project demo pytest demo/tests
uv run --project demo ruff check demo
uv run --project demo ruff format --check demo
uv run --project demo mypy --config-file demo/pyproject.toml demo/marimba_demo
npm --prefix demo/frontend ci
npm --prefix demo/frontend run lint
npm --prefix demo/frontend run typecheck
npm --prefix demo/frontend test
npm --prefix demo/frontend run build
python3 scripts/check_public.py
```

The server protocol tests use a contract double, not a live demo model. Frontend
unit tests cover threshold crossing, five-second trails, out-of-order/duplicate
updates, p95 and binary encoding. No fake-model launcher is shipped.
Scheduler tests block a synthetic decoder while live windows advance, verify
bounded work and preserved capture provenance beyond timestamp-history eviction,
and require the full longer context before the first buffered review.

For an installable wheel, run the frontend build before `uv build --project demo`.
The wheel includes the built frontend; users then need only Python and the
inference engine, not Node. The source-checkout launcher builds on first use.

## Public audio and attribution

Use only audio you have permission to redistribute. For speech, [LibriSpeech,
OpenSLR 12](https://www.openslr.org/12) is CC BY 4.0; credit the corpus authors
(Panayotov, Chen, Povey, Khudanpur) and note any trimming/resampling. One suitable
music source is [Bach's Prelude and Fugue in C major, BWV 846, Musopen recording
on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:JS_Bach_-_Well_Tempered_Clavier_-_Prelude_in_Fugue_No.1_in_C_major.ogg),
identified there as CC0. The composition's public-domain status alone would not
establish the recording's license. No audio or model weights are committed here.

IBM Plex font files are included under the SIL Open Font License in
`frontend/public/OFL.txt`; the supplied Marimba mark is used unchanged.
