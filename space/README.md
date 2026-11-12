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

Multilingual typed audio decisions with upload, microphone recording, six typed
questions, a user-supplied term, and JSON probabilities. The app calls
`Marimba.from_pretrained` and `Question`; it does not implement a separate inference
engine. Term scoring remains greedy decoding plus four beam hypotheses.

From the source checkout root:

```sh
pip install -r space/requirements.txt
python space/app.py
```

For local CPU use, set `MARIMBA_DEVICE=cpu`. The default is CUDA float32. The model
is loaded once at startup; GPU inference is wrapped in `@spaces.GPU(duration=120)`.
Requests run one at a time. Uploads are limited to 25 MB and the queue to 20 requests.

## Private deployment

The package is imported from bundled source, not PyPI or a private Git dependency.
The deployment script uploads only the package, licenses, build configuration,
and Space files. It copies `space/requirements.txt` to the root requirements file
and uses this file as the Space card. The app imports the bundled `src/` directly
because Spaces installs dependencies before copying application files. It never
uploads a token or environment file. The Spaces builder manages the `spaces`
runtime version.

Authenticate locally with `HF_TOKEN`, then run from the checkout root:

```sh
python space/deploy.py --repo-id charcot-ai/marimba
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
selecting paid hardware. The script never silently upgrades hardware.

## Questions and limitations

- Speech: `Question.noul("Is someone speaking?", seconds=2.0)`, using the package's
  validated two-second tail and default 0.225 threshold; the Space does not tune it.
- Turn ended: `Question.score("Has the speaker finished?")`.
- Language: English, Portuguese, Spanish, or Other (all remaining supported languages).
- Term: a user-supplied word or phrase, matched by the package's exact-token scorer.
- Speaker change: dominant speaker in the last two seconds versus the prior six.
- People: speaker count, with choices 0, 1, 2, and 3+; not a count of everyone present.

Only the last eight seconds are scored, not an entire recording. Speaker change
and people count require eight valid seconds; partial windows return no prediction.
Term probabilities are **uncalibrated fractions of five hypotheses**. Decoder
context above eight seconds is unvalidated; no real-time guarantee is claimed.

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
