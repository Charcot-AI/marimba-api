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

- Speech: `Question.noul("Is someone speaking?")`.
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

Three eight-second, mono 16 kHz PCM clips are generated with eSpeak NG 1.50:
English containing “marimba,” Portuguese containing the near-word “marina” but
not “marimba,” and a Spanish exchange using two different synthetic voices. These
are functionality examples, not an accuracy or calibration benchmark. There are
no human recordings. The text and synthesis settings are in `generate_examples.py`.

Regenerate with eSpeak NG and FFmpeg installed:

```sh
python space/generate_examples.py
```

The clips are dedicated to the public domain under CC0-1.0 to the extent permitted
by law. eSpeak NG is used as a synthesis tool; no voice-model files are distributed.

## API

```python
import os
from gradio_client import Client, handle_file

client = Client("charcot-ai/marimba", token=os.environ["HF_TOKEN"])
decisions = client.predict(
    handle_file("space/examples/english_marimba.wav"),
    ["Speech", "Turn ended", "Language", "Term", "Speaker change", "People"],
    "marimba",
    api_name="/decide",
)
```

Run `python space/smoke_test.py` for all six question types and the three synthetic
examples. It reports client wall time (including queue and transfer) separately
from the package's decision-lane latency, and writes JSON only when `--output` is
provided. Tokens are never part of its output.

Do not upload sensitive recordings. Gradio caches uploads temporarily; this is
not a privacy, retention or deletion guarantee. A deployed service needs its own
authentication, consent, retention controls and operational review.
