# Marimba live instruments

A loopback-only app for microphone or file audio, with parallel decision cards.
The browser sends mono PCM16 at 16 kHz; the server calls the inference package's
`model.stream(questions, window_seconds=8, hop_seconds=0.25, sample_rate=16000,
coalesce=True)`. Its cheap `ingest()` retains every packet in the ring buffer;
`decide()` snapshots the latest window and runs inference in a worker thread.
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

Installing both editable projects together preserves the demo's dependency constraints
when the inference engine resolves its optional dependencies. `--no-sync` then
preserves the separately installed inference engine. The frontend uses
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
the last decisions, which dim after 2.5 seconds without a new result. Restart clears
the previous session, its trail, and timing samples. Inputs are not auto-reconnected.

## Cards and visual language

The checkpoint's `model.capabilities` is authoritative. Unavailable tasks are
omitted, not drawn with invented answers. Experimental tasks are explicitly marked.
Uncalibrated term support is labelled **UNCALIBRATED / RAW N-BEST**, not presented
as a calibrated probability. The current language card is a closed set
of the ISO codes `en`, `pt`, `es`; it does not claim an open-set "Other" detector.
Depending on the checkpoint, this can be fewer than the eight candidate cards:

| Task | Instrument |
|---|---|
| Turn completion | 0–1 ruler, pointer, trailing positions from the last five seconds |
| Speech present | P(yes) needle and a yes/no reading |
| Speaker change | P(yes) needle; experimental when advertised |
| Same speaker | P(yes) needle; experimental when advertised |
| Language | One thin horizontal probability bar per option |
| Sound scene | Speech/music/noise bars, only if available |
| Heard marimba | Support needle; explicitly labelled when uncalibrated |
| Speakers | Per-option probability bars; experimental when advertised |

Position on a shared scale and aligned bar lengths make small comparisons easier
than interpreting colored areas or angles. See Cleveland and McGill,
[Graphical Perception (1984)](https://doi.org/10.1080/01621459.1984.10478080).
This motivates the linear rulers and VU-style bars, rather than decorative dials.
Numbers and labels remain visible; nothing depends on color alone. A binary/score
threshold crossing briefly accents one hairline without a color flash. Motion
respects the system reduced-motion preference. The UI follows the Charcot kit:
IBM Plex, thin geometry, a grid, black/white or warm white, no bold, shadows or gradients.

## Timing: measure on your own machine

There are two different measurements. Do not compare them as if they were the same:

- **Per-card model ms:** the package's wall-clock hop inference duration, including
  encoder, heads, lexical lane and accelerator synchronization. Capture, resampling,
  socket round trip and rendering are excluded. Parallel cards may share this number.
- **Capture → render p95:** nearest-rank p95 over the last 60 seconds of received
  decision updates. It includes the 250 ms packetization interval, socket round trip,
  inference and next animation frame. Capture start is estimated from the worklet
  audio timestamp minus 250 ms; microphone hardware/OS buffering is not directly measured.
  Worklet delivery is mapped to the AudioContext timeline, so main-thread queueing
  is included. This is an app-observed estimate, not a hardware-loopback acoustic measurement.

The ≤300 ms end-to-end target is a target, not a guarantee. A missed target is
labelled explicitly. Startup/model download is excluded; allow several warmup hops.
Only one inference call runs at a time. Audio ingest continues while it computes,
so slow inference never repeatedly resets the context or concatenates separated
audio as if continuous. The footer reports the actual spacing of scored windows
and counts coalesced decision ticks, not dropped input. There is no promise of a
decision at every 250 ms hop on slower hardware. If audio transport itself backs
up, the app stops explicitly instead of silently losing audio. A missing or
duplicate sequence closes the session. Socket queues and timestamp history are bounded.

For a lower-load CPU subset without lexical decoding, start with
`marimba-demo --device cpu --tasks turn_end,language` (using the
same `uv run --project demo --no-sync` prefix). Unsupported tasks are still
omitted. This changes the workload; do not compare its timing to a full-grid run
without listing the different questions.

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
