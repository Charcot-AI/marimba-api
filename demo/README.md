# Marimba live instruments

A loopback-only app for microphone or file audio, with parallel decision cards.
The browser sends continuous mono PCM16 at 16 kHz in 250 ms packets. The server
retains audio for active windows and utterances, without saving audio or transcripts.
An optional user-installed transcription adapter can send audio to another service;
check that service's retention policy separately.

## Run locally

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22 LTS and npm.
Use Chrome or Safari with microphone access allowed for localhost. The inference
package and compatible model weights must be available; this app never substitutes
fabricated model outputs for missing weights.

From the repository root, install once:

```sh
uv sync --project demo --python 3.12
uv pip install --python demo/.venv/bin/python -e '.[torch]' -e demo
```

Then one command builds the frontend on first launch, loads the model,
starts Python on `127.0.0.1:8765`, and opens the page:

```sh
uv run --project demo --no-sync marimba-demo
```

Installing both editable projects from this repository root uses the in-repo
inference engine and preserves the demo's dependency constraints. `--no-sync`
keeps that combined installation intact.

For reproducibility, pass `--revision <model-commit>` and record the runtime shown
at the bottom of the page. `--device cpu` selects CPU; `--device auto --backend auto`
lets the package select a supported accelerator. Apple Silicon acceleration is
not guaranteed: check `marimba doctor` and the displayed device. CPU/Linux timings
cannot be relabelled as Apple GPU results. `--no-open` suppresses opening the
browser; `--port 8766` uses another loopback port.

Click **Start microphone** or **Open audio**. File audio is decoded locally,
downmixed, played audibly and paced in real time through the same worklet/socket.
If a browser rejects a 16 kHz AudioContext, the worklet resamples its native rate.
Use headphones to avoid microphone feedback. Restart clears decisions, trails,
timing and drop counters. Inputs are not auto-reconnected.

## Windows and cadence

The Speech card scores Marimba's voice-activity head on the final 2 s of each
rolling 8 s window. Its yes/no flips are adjusted by the measured readout delays:
onset is estimated 0.5 s before the yes window end, and offset 1.75 s before the
no window end. These are timing estimates, not word boundaries or a latency
guarantee. The timeline stores sample positions and includes an open run through
the last observed window end for coverage checks.

| Card | Audio window | Cadence |
|---|---|---|
| Turn ended (yes/no) | Up to 8 s ending at the estimated speech offset; left padding masked False | At end of speech |
| Speech (last 2 s) | Final 2 s of the rolling 8 s window | Every 250 ms, coalesced under load |
| Speaker changed | Full rolling 8 s, including internal silences | Every 250 ms when eligible |
| People (last 8 s) | Full rolling 8 s, unchanged model count | Every 250 ms, coalesced under load |
| Language | Each closed speech run, split into consecutive clips of at most 8 s | Per utterance/chunk |
| Marimba spoken | Same utterance chunks as language | Per utterance/chunk |

Turn ended uses the Platt probability with the default 0.5 threshold; no fitted
event threshold is available. The estimated speech offset does not guarantee
that every detected speech run ends a conversational turn.

The 2 s tail readout uses temperature 0.7869027171235091 and threshold 0.225.
Held-out results are AUROC 0.998, F1 0.993, and specificity 0.967 at that
threshold. People counts identities in the last 8 s, not the whole conversation,
and is never forced to zero by speech detection.

Speaker change requires a full mask and at least **0.5 s** of estimated speech in
**each** of the window-relative intervals [0,6) and [6,8). Otherwise the card shows
the previous answer pending, with a shimmer on its value only. Before its first
eligible answer it shows a dash. This is a necessary coverage rule, not a
guarantee of a single dominant identity on either side. Silences remain in the
model input.
Decoder questions use audio at the **start** of the clip with zero padding at the
**end**, matching their evaluation alignment. Real internal silences are retained.
No wall-clock review pause or longer review context is used. Hard 8 s chunk
boundaries can split words; overlap and word alignment are not claimed here.
Workers process utterances FIFO. While one utterance is running, all waiting
utterances are retained in order and counted in the footer. Active utterance
chunks finish in order. No input audio is dropped from the live timeline by this
decoder backpressure policy.

Each card shows its actual audio span, model time, observed update interval,
received age and audio age. Turn/decoder results remain displayed until the next
eligible event, rather than being replaced by rolling-window answers. Graceful
Stop drains queued closed utterances and estimated offsets (up to a 90 s browser timeout).
An open utterance is not treated as closed just because capture stops.
A positive Term result is shown for 2 seconds, then the card returns to its empty state.
Disconnect discards subsequent results; an in-flight model call must still finish
before another session can own the model.

Ineligible speaker updates do not replace the last value, trail or audio-window
footer. Only the reading shimmers while that update is pending, while a decoder
is computing, or when a rolling result is stale. The rest of the card is not
dimmed. The shimmer follows the selected theme and stops with reduced motion.

One rolling encoder worker, one turn worker and one decoder worker run separately.
The model serializes encoder passes but does not hold the encoder lock during
lexical decoding. Slow inference coalesces rolling decisions, not input packets.
The footer counts skipped rolling ticks separately from queued decoder utterances.
If transport backs up, the app stops instead of silently losing continuity.
A missing/duplicate sequence closes the session. The server accepts only loopback
hosts and same-origin WebSockets. Do not publish it through a public tunnel.

## Cards and visual language

`model.capabilities` is authoritative. Unavailable tasks are omitted; experimental
tasks are marked. The six cards share one responsive grid. Term support is labelled
**UNCALIBRATED / RAW N-BEST**; it is not a calibrated probability. Language is a
closed set of `en`, `pt`, `es`, not an open-set detector.

Linear rulers and aligned probability bars follow Cleveland and McGill,
[Graphical Perception (1984)](https://doi.org/10.1080/01621459.1984.10478080).
Numbers and labels remain visible; nothing depends on color alone. Threshold
crossings briefly accent one hairline. Reduced-motion preferences are respected.
The UI uses IBM Plex, thin geometry, a black/white or warm-white grid, and no
bold, shadows or gradients. All assets are local; no CDN or analytics is used.

For a lower-load CPU subset, pass `--tasks turn_end,voice_activity`. This changes
the workload; do not compare timings with a full-grid run without listing tasks.
Set thread counts from measurements, for example `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.
Per-card model time excludes capture, transport and paint; it includes the relevant
encoder call. CPU decoding can lag well behind real time. Neither a 250 ms ingest
hop nor the speech readout's flip delay guarantees a completed model decision every hop.

## Units, speaker cuts and speakers for a transcription pipeline

`marimba_demo.pipeline.run_pipeline` replays audio hop by hop through the public
stateless calls (`Marimba.decide_windows` and `Marimba.decide_pairs`, the same
questions the [window server](../docs/window-server.md) answers) and reads only audio
before the current hop. It does not change the live cards. Every number below is a
field of `UnitConfig`, `CutConfig` or `PipelineConfig`.

**Speech units** (`units.py`). The signal is the 1 s `voice_activity` probability, one
per hop. A segment opens on the first hop with `p >= 0.1` and closes on the first hop
with `p < 0.1`; there is no hysteresis or hangover. Its start is the opening window end
minus 0.5 s and its end is the closing window end minus 0.75 s; segments under 0.25 s
are dropped. Segments join into one unit while the previous segment did not end a turn
(`turn_end` asked on the eight seconds ending at that segment's end), the pause is at
most 5 s and the unit stays within 30 s; longer speech is sliced at 30 s. Each unit edge
then moves to the quietest 10 ms frame within 0.5 s, bounded by the neighboring units:
mean-square energy, centered five-frame average, in dB rounded to 1 dB, ties to the
frame nearest the original edge. The edge is placed at the center of that frame. The
audio handed to transcription is padded by 0.25 s on each side, capped at the midpoint
of the gap to the neighboring unit and at the audio bounds; unit edges stay snapped.
`snap_units` applies this rule to a finished list. `run_pipeline` emits each unit as
soon as it closes, so a later unit may not exist yet: the end padding is then capped by
the audio heard so far, and the next unit's start padding never crosses it.

**Speaker cuts** (`speakers.py`). After transcription, `speaker_change` is asked at
word ends at least 0.5 s apart, on the 6 s before and the 2 s after each. Scores of 0.5
or more are kept strongest first and at least 2 s apart, and each is moved to the
nearest sentence end within 1.5 s. A peak with no sentence end in reach cuts at the
nearest word end (`word_fallback=False` drops it instead).

**Primer bank** (`speakers.py`). `PrimerBank` starts empty. Each span is compared with
every stored primer through `same_speaker`; the speaker with the highest mean score
takes the span at 0.5 or more, otherwise a new speaker opens until there are four.
A speaker keeps at most three primers of up to 4 s, taken from matched spans of at
least 2 s; the first span of a new speaker is always kept.

### Validation at a 500 ms hop

`python -m marimba_demo.validate` (needs the `datasets` extra) builds a 182 s
conversation from public LibriSpeech `validation.clean` utterances of four readers with
known pauses of 0, 0.4, 0.8, 1.5 and 6 s, and compares the pipeline with the known
speech edges and speakers. Word times are synthetic: each utterance's words are spread
over its speech by character count and its last word ends the sentence. One run on
CPU, float32:

| Measure | Result |
|---|---|
| Opening hop after a known onset (assumed 0.5 s) | mean 0.68 s, median 0.58 s, 11 onsets |
| Closing hop after a known offset (assumed 0.75 s) | mean 0.94 s, median 0.85 s, 11 offsets |
| Unit start error, before and after snapping | median 0.17 s, then 0.025 s (largest 1.29 s) |
| Unit end error, before and after snapping | median 0.22 s, then 0.075 s (largest 0.79 s) |
| Known speech left outside units | 5.5 s raw, 1.8 s snapped, 1.3 s with padding |
| Start snapped at the opening hop against the final snap | identical for all 17 units |
| Audio buffered before the opening hop to pad a start | at most 1.245 s |
| Speaker changes inside units found within 1 s | 5 of 5 |
| Cuts placed | 18, of which 5 at a known change and 5 at a sentence end |
| Primer bank | 4 speakers for 4 readers; 99.6% of the known speech inside spans on the right speaker |

Reading the table: at a 500 ms hop both readout flips arrive about 0.18 s later on
average than the two lag constants assume, and snapping absorbs most of it. Thirteen
of the eighteen cuts were not at a speaker change and thirteen used the word-end
fallback; whether those are the same thirteen was not checked. The extra cuts cost
little here: the primer bank still put 99.6% of the speech on the right speaker. Every
unit in this run ended with a turn end, so joining across a pause is covered by the
synthetic tests only.

## Optional transcription

No adapter is shipped or enabled by default. The transcript panel stays hidden.
The interface is `frontend/src/stt.ts`: `start(onTranscript, onError)`, `push(pcm,
capturedAtEpochMs)`, `stop()`. Transcript events have stable IDs and may replace
partial IDs. Text is inserted as text nodes, never HTML.

Place your adapter exporting `createAdapter(config)` at `demo/local/stt_adapter.ts`.
The launcher compiles it to `local/stt_adapter.js` outside the public bundle.
`local/` is gitignored and never packaged. Set `MARIMBA_STT_CONFIG` to a local JSON
object; the browser fetches it from a no-store, same-origin loopback endpoint.
Never commit credentials or an adapter implementing a non-public protocol.

## Development and checks

After the joint installation above:

```sh
uv run --project demo --no-sync pytest tests demo/tests
uv run --project demo --no-sync ruff check demo
uv run --project demo --no-sync ruff format --check demo
uv run --project demo --no-sync mypy --config-file demo/pyproject.toml demo/marimba_demo
uv run --locked mypy src
npm --prefix demo/frontend ci
npm --prefix demo/frontend run lint
npm --prefix demo/frontend run typecheck
npm --prefix demo/frontend test
npm --prefix demo/frontend run build
python3 scripts/check_public.py
uvx pre-commit run --all-files
```

Unit tests use synthetic audio and contract doubles, not live inference. They cover
coverage eligibility, exact offset windows, consecutive utterance chunks,
decoder end padding, latest-pending drops and capture provenance. Frontend tests
cover neutral readings, event cadence, timing, duplicate/out-of-order updates,
threshold trails, binary encoding and production assets.

Build the frontend before `uv build --project demo` for an installable wheel.
The wheel includes built assets; users then need Python and the inference engine,
not Node. The source-checkout launcher builds on first use.

## Public audio and attribution

Use only audio you may redistribute. [LibriSpeech, OpenSLR 12](https://www.openslr.org/12)
is CC BY 4.0; credit Panayotov, Chen, Povey and Khudanpur, noting trimming/resampling.
One music source is [Bach BWV 846, Musopen recording on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:JS_Bach_-_Well_Tempered_Clavier_-_Prelude_in_Fugue_No.1_in_C_major.ogg),
identified there as CC0. Public-domain composition status alone does not establish
the recording's license. No audio or model weights are committed here.

IBM Plex files are included under the SIL Open Font License in
`frontend/public/OFL.txt`; the supplied Marimba mark is used unchanged.
