---
title: Marimba
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: space/app.py
python_version: "3.10.13"
private: true
---

# Marimba Space

Multilingual typed audio decisions using the local demo's cards, questions and
Scheduler. Record or upload up to 30 seconds. The default ZeroGPU mode replays
every 250 ms hop in one GPU lease, waiting for each rolling encoder update
without coalescing. The final partial frame is zero-padded to 250 ms. Queued turn
and utterance work drains before returning. Every replay appends the same trailing
silence: `OFFSET_LAG_S` plus one hop, rounded up to whole frames (currently 2 s).
This lets the last utterance close as in a live stream, without inspecting outputs.
Each event includes input, frame-padding and trailing-silence durations in `replay`
metadata. The eight example files remain unchanged.

The UI shows final card values, timed utterance results, a rolling probability
plot and collapsed raw events. The People line is the selected count's probability,
not the number of people. The app calls `Marimba.from_pretrained` and the demo's
`available_cards` / `questions_for`; it does not implement separate inference.
Term scoring remains greedy decoding plus four beam hypotheses.

From the source checkout root:

```sh
pip install -r space/requirements.txt
python space/app.py
```

For local CPU use, set `MARIMBA_DEVICE=cpu`. The default is CUDA float32. The model
is loaded once at startup; GPU inference is wrapped in `@spaces.GPU(duration=120)`.
Replay requests run one at a time. Uploads are limited to 25 MB and the queue to 20 requests.

## Private deployment

The package is imported from bundled source, not PyPI or a private Git dependency.
The deployment script uploads only the package, licenses, build configuration,
Space files, and the demo's `__init__`, `cards`, `protocol`, `scheduler`, `speech`
and `replay` modules (not its server, CLI or static assets). It copies
`space/requirements.txt` to the root requirements file
and uses this file as the Space card. The app imports the bundled `src/` directly
because Spaces installs dependencies before copying application files. It never
uploads a token or environment file. The Spaces builder manages the `spaces`
runtime version.

Authenticate locally with `HF_TOKEN`, then run from the checkout root:

```sh
python space/deploy.py --repo-id your-account/your-private-space
```

The script creates a **private** Gradio Space on ZeroGPU, refuses to update a public
Space, and configures `HF_TOKEN` as a Space secret for authenticated model downloads.
The `private` card field is descriptive metadata, not access control. The API
creation flag and the verified repository visibility enforce privacy. This script
does not change model or source-repository visibility, publish a package, or merge
code. A later public launch requires a separate, explicit visibility change.

ZeroGPU allocates GPU time only during inference and releases it afterward; no
dedicated paid GPU is kept idle. If ZeroGPU is unavailable, select L4 explicitly
with `--hardware l4x1`; the script sets a 300-second idle sleep time. Consult the
[current hardware pricing](https://huggingface.co/docs/hub/spaces-gpus) before
selecting paid hardware. The script never silently upgrades hardware and refuses
to update a public Space.

## Questions and limitations

| Card | Audio window | Cadence |
|---|---|---|
| Turn ended | Up to 8 s ending at estimated speech offset; masked left padding | At end of speech |
| Speech (last 2 s) | Final 2 s of rolling 8 s | Every 250 ms |
| Speaker changed | Full rolling 8 s, with speech coverage required in both parts | Every 250 ms when eligible |
| People (last 8 s) | Full rolling 8 s, unchanged count (0 / 1 / 2 / 3+) | Every 250 ms |
| Language | Closed speech runs, split into consecutive clips of at most 8 s | Per utterance/chunk; en / pt / es |
| Term | Same utterance chunks as Language | Per utterance/chunk; `Was "{term}" spoken?` |

Speech uses the validated 0.225 threshold. The demo estimates onset 0.5 s before
the first yes window end and offset 1.75 s before the first no window end.
Replay does not change those estimates, prompts, thresholds or windows. Recordings
with no detected speech can leave utterance cards pending.
Speaker tasks require eight valid seconds; partial windows return no prediction.
Term probabilities are **uncalibrated fractions of five hypotheses**. Decoder
context above eight seconds is unvalidated; no real-time guarantee is claimed.

## Optional live mode

Only `MARIMBA_LIVE=1` enables a streaming microphone. Incoming chunks are downmixed,
continuously resampled to 16 kHz and buffered into 4000-sample frames. Each session
owns a Scheduler in Gradio state; its live/turn/review tasks run on Gradio's loop.
Rolling updates coalesce under load, as in the demo. One live session owns the
model at a time; other sessions receive a busy error. Stop drains queued work;
disconnect/state deletion aborts and releases the session. Live history is bounded.
Live mode never appends silence or pads its incomplete final frame.

`deploy.py --hardware l4x1` sets `MARIMBA_LIVE=1`; default ZeroGPU deployment removes
it. Never select paid hardware without approval. To exercise the same mode locally:

```sh
MARIMBA_DEVICE=cpu MARIMBA_LIVE=1 python space/app.py
```

## Synthetic examples

Eight examples are **synthetic speech generated with ElevenLabs**, using the
`eleven_v4` model and premade Sarah, George, and Bella voices. The source owner
confirmed generation on a paid plan permitting commercial use. There are no real
recordings, and no ElevenLabs API token or voice-model weights are distributed.
The resulting audio is not being relicensed as CC0 or Apache-2.0.

The examples cover a completed turn, a cut-off turn, “marimba,” the near-words
“marina/mariscos/maraca,” two speakers, Spanish, Portuguese, and three voices.
Transcripts, source sample intervals, source SHA256, and clip hashes are recorded
in `examples/manifest.json`. All eight scenes are included without selecting by
model output. These are functionality examples, not an accuracy benchmark.

Every clip ends at the final speech segment's end plus 0.3 seconds, using the
source manifest's sample timings. The last eight seconds are retained; shorter
scenes are left-padded with digital silence so speaker tasks receive a full
eight-second window. The same rule applies to every scene, without selecting
timings from model predictions.

Reproduce the cuts from the original assembly and its timing manifest:

```sh
python space/generate_examples.py /path/to/scenes.wav /path/to/manifest.json
```

The script verifies the source SHA256 before cutting. It does not call a TTS API,
alter levels, or require the original paid synthesis credentials. The original
assembly is not needed to run the Space; the eight derived clips are bundled.

## API

```python
import os
from gradio_client import Client, handle_file

client = Client("charcot-ai/marimba", token=os.environ["HF_TOKEN"])
events = client.predict(handle_file("space/examples/03_term.wav"), "marimba", api_name="/replay")
# Backward-compatible one-shot endpoint, not the demo replay path:
decisions = client.predict(
    handle_file("space/examples/03_term.wav"),
    ["Speech", "Turn ended", "Language", "Term", "Speaker change", "People"],
    "marimba",
    api_name="/decide",
)
```

Run `python space/smoke_test.py` for all six question types and the eight synthetic
examples. It reports client wall time (including queue and transfer) separately
from the package's decision-lane latency, and writes JSON only when `--output` is
provided. Tokens are never part of its output.

Do not upload sensitive recordings. Gradio caches uploads temporarily; this is
not a privacy, retention or deletion guarantee. A deployed service needs its own
authentication, consent, retention controls and operational review.
