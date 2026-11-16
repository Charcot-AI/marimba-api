<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Charcot-AI/marimba-api/main/docs/assets/marimba-horizontal-white.svg">
    <img src="https://raw.githubusercontent.com/Charcot-AI/marimba-api/main/docs/assets/marimba-horizontal-black.svg" width="320" alt="marimba">
  </picture>
</p>

<p align="center">
  Typed, calibrated decisions about short and streaming audio.<br>
  <a href="https://huggingface.co/charcot-ai/marimba">Model</a> ·
  <a href="https://huggingface.co/spaces/charcot-ai/marimba">Space</a> ·
  <a href="docs/index.md">Docs</a> ·
  <a href="https://pypi.org/project/marimba-audio/">PyPI</a> ·
  <a href="https://charcot.ai">charcot.ai</a>
</p>

## What it is

Marimba answers typed questions about the last eight seconds of an audio window.
One pass of a frozen [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
encoder serves every question; small readouts, a question-conditioned head and the
frozen decoder turn it into answers. It is not speech-to-text and does not produce
speaker identities or timelines.

| Kind | Constructor | `value` |
|---|---|---|
| `noul` (yes/no) | `Question.noul("Is someone speaking?")` | `bool` |
| `score` | `Question.score("Has the speaker finished?")` | probability as `float` |
| `choice` | `Question.choice("What language?", ["en", "pt"])` | chosen option, plus `probabilities` |

This repository holds the `marimba-audio` Python package, the OpenAI-compatible
server, voice-framework integrations, the local demo, the Space and the benchmarks.

## Install

```sh
pip install "marimba-audio[torch]"
```

Python 3.10 or newer. The distribution is `marimba-audio`; the import is `marimba`.
The core install has no inference engine, so add the `torch` extra to run the model.

| Extra | Adds |
|---|---|
| `torch` | Torch backend for the pretrained lanes |
| `datasets` | Hugging Face `datasets` for `map_fn` |
| `server` | FastAPI and Uvicorn for `marimba serve` |
| `pipecat` | Pipecat turn analyzer (Python 3.11+) |
| `livekit` | LiveKit Agents turn detector |
| `demo` | Gradio |
| `onnx` | ONNX Runtime discovery only; the pretrained lanes are torch-only |

## Quickstart

```python
from huggingface_hub import hf_hub_download
from marimba import Marimba, Question

audio = hf_hub_download("charcot-ai/marimba", "space/examples/01_turn_end.wav", repo_type="space")
model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
questions = [
    Question.score("Has the speaker finished?"),
    Question.noul("Is someone speaking?"),
    Question.choice("What language?", ["en", "pt", "es"]),
    Question.noul('Did someone say "everything"?'),
]
for decision in model(audio, questions):
    print(decision.task, decision.kind, decision.value, decision.probability)
```

The example clip is synthetic speech from the Space. The first load downloads the
weights at the revision pinned in `marimba.weights.DEFAULT_MODEL_REVISION`, with
SHA256-verified files, plus the pinned Qwen3-ASR-0.6B encoder.

To build an integration without downloads, use the deterministic synthetic model.
Its outputs are not model predictions.

```python
import numpy as np
from marimba import Question
from marimba.testing import FakeMarimba

model = FakeMarimba()
questions = [Question.noul("Is someone speaking?")]
decisions = model(np.zeros(16000, dtype=np.float32), questions, sampling_rate=16000)
print(decisions[0].to_dict())
```

## Questions

| Task | Example question | Kind | Status | Evidence |
|---|---|---|---|---|
| `turn_end` | `Has the speaker finished?` | `score`, `noul` | validated | AUROC 0.930 on Smart Turn v3.2 evaluation; 0.716 out of domain |
| `voice_activity` | `Is someone speaking?` | `noul` | validated | AUROC 0.995 on AMI, 0.9996 on VoxConverse; default threshold 0.225 |
| `language` | `What language?` + options | `choice` | validated | Native LID, 30 languages; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| `term` | `Did someone say "hello"?` | `noul` | validated | Raw fraction of greedy + four beam hypotheses; uncalibrated |
| `same_speaker` | `Is this the same speaker?` | `noul`, `score` | validated | AUROC 0.998 on AMI, 0.996 on VoxConverse |
| `speaker_change` | `Did the speaker change?` | `noul`, `score` | validated | Last 2 s vs previous 6 s; AUROC 0.966 on AMI, 0.980 on VoxConverse |
| `speaker_count` | `How many speakers?` + `0`, `1`, `2`, `3+` | `choice` | validated | Accuracy 0.778 on AMI, 0.859 on VoxConverse; undercounts 3+ |
| `pii` | `Does this audio contain any personal information?` | `noul` | experimental | Needs a caller-supplied text detector; none included |
| `audio_class` | | `choice` | unavailable | No installed lane |

Unavailable tasks return `None` values, never a fallback prediction. Evidence,
routing rules and option rules: [capabilities](docs/capabilities.md).

### Compared with audio LLMs

Paired comparison on the same 1,000 fixed public questions. The audio LLMs answer zero-shot, scoring each option by summed log-probability. Choice rows use accuracy; binary rows use AUROC.

| Slice | n | Metric | Marimba | Qwen3-Omni-30B-A3B | Qwen2-Audio-7B | FireRedAudio |
|---|---:|---|---:|---:|---:|---:|
| Common Voice language | 52 | accuracy | 0.904 | 1.000 | 0.846 | 0.962 |
| FLEURS language | 200 | accuracy | 1.000 | 1.000 | 0.840 | 0.865 |
| Common Voice term, choice | 41 | accuracy | 0.976 | 0.976 | 0.829 | 1.000 |
| FLEURS term, choice | 62 | accuracy | 0.855 | 0.855 | 0.758 | 0.871 |
| Speech Commands, choice | 60 | accuracy | 0.933 | 1.000 | 0.967 | 0.950 |
| Common Voice term, binary | 107 | AUROC | 0.907 | 0.858 | 0.802 | 0.836 |
| FLEURS term, binary | 138 | AUROC | 0.901 | 0.791 | 0.682 | 0.776 |
| Speech Commands, binary | 140 | AUROC | 0.926 | 0.992 | 0.940 | 0.993 |
| Turn end (Smart Turn v3.2) | 200 | AUROC | 0.935 | 0.786 | 0.476 | 0.542 |

- Marimba's turn-end readout is trained on Smart Turn training data, so that row is in-distribution for Marimba only. All other rows are zero-shot for every system.
- [FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio) was added after the [paper](https://arxiv.org/abs/2611.18330), which reports the other three systems. It received the same Portuguese prompts as the others, while its understanding path is trained on Chinese and English. It ran in bf16 without its optional optimized attention kernels.

## Outputs

Each question returns one `Decision`; `Decision.to_dict()` is JSON-safe.

| Field | Meaning |
|---|---|
| `task`, `kind`, `question` | Routed task, answer kind and the question asked |
| `value` | `bool` for `noul`, `float` for `score`, option for `choice`; `None` when unavailable |
| `probability` | Positive-class or chosen-option probability |
| `probabilities` | Per-option probabilities for `choice` |
| `score`, `threshold` | Raw lane score and the threshold `noul` compares against |
| `status` | `validated`, `experimental` or `unavailable` |
| `calibrated` | `True` when a [domain calibration](docs/calibration.md) is attached |
| `window_start_s`, `window_end_s` | Audio time the decision describes |
| `latency_ms` | Synchronized inference time, excluding loading and decoding |

## Use it

| Need | Entry point | Guide |
|---|---|---|
| Files, bytes, arrays | `model(audio, questions)` | [Audio inputs](docs/audio.md) |
| Many clips | `model.batch(audios, questions)` | [Batch](docs/batch.md) |
| Hugging Face datasets | `ds.map(model.map_fn(questions), batched=True)` | [Batch](docs/batch.md) |
| Live audio | `model.stream(questions)` with `push`/`flush` or `ingest`/`decide` | [Streaming](docs/streaming.md) |
| Split encoder and decoder work | `model.encode`, `decide_encoded`, `decide_snapshot` | [Scheduling](docs/scheduling.md) |
| Your own domain | `model.calibrate(windows, labels, question)` | [Calibration](docs/calibration.md) |
| HTTP for routers and OpenAI SDKs | `marimba serve` | [Server](docs/openai-server.md), [hosting](docs/hosting.md) |
| Pipecat | `MarimbaTurnAnalyzer` | [Example](examples/pipecat/README.md) |
| LiveKit Agents | `MarimbaTurnDetector` | [Example](examples/livekit/README.md) |
| Vapi | Custom transcriber bridge | [Example](examples/vapi/README.md) |
| Local live demo | `marimba-demo` | [Demo](demo/README.md) |
| Runpod Serverless | Decoder worker | [Worker](.runpod/README.md) |

Streaming takes mono 16 kHz chunks; each complete hop emits one update:

```python
stream = model.stream(questions, hop_seconds=0.25)
updates = stream.push(np.zeros(4000, dtype=np.float32))
updates += stream.flush()
print([update.to_dict() for update in updates])
```

## Runtime and latency

`marimba doctor` reports platform, extras, devices and ONNX providers.
`marimba bench --fake --questions 4 --hops 100 --warmup 10` times the integration
path with synthetic outputs. The pretrained lanes run on torch CPU or CUDA; CUDA
`dtype="auto"` picks bf16, then fp16, then fp32. MPS is opt-in and not validated.
See [runtime](docs/runtime.md).

Public-API calls on one NVIDIA L4, bf16, batch one, eight-second windows:

| Question | p50 ms | p95 ms |
|---|---:|---:|
| Turn end | 12.23 | 13.31 |
| Voice activity | 14.40 | 15.77 |
| Language | 153.67 | 160.80 |
| Term | 1399.44 | 2349.14 |

Protocol, CPU and Mac numbers: [benchmarks](docs/benchmarks.md). No real-time
guarantee is claimed.

## Limits

- Decisions describe the last eight seconds; shorter audio is left-padded and masked.
- Decoder context above eight seconds is **unvalidated**, although the API accepts up to 30 seconds.
- Speaker questions need eight valid seconds and return `None` on partial windows.
- Term probabilities are uncalibrated fractions of five hypotheses.
- `Other` in language choices aggregates the unlisted supported languages; it is not an unknown-language detector.
- Accent, language, noise, overlap, microphone and domain shifts can change decisions; [calibrate](docs/calibration.md) on your own labels.
- Not for diagnosis, identity verification, surveillance or safety-critical automation.

## Development

```sh
uv sync --locked --extra datasets --extra pipecat --extra server --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Real-model tests download weights and are excluded by default:
`uv run pytest -m real_model`. See [CONTRIBUTING](CONTRIBUTING.md) and
[CHANGELOG](CHANGELOG.md).

## Citation

Cite the paper:

```bibtex
@misc{amadori_marimba,
  author = {Solano Amadori},
  title = {Marimba: A System One Model for Speech},
  archivePrefix = {arXiv},
  eprint = {[PENDING]},
  url = {https://huggingface.co/charcot-ai/marimba}
}
```

To cite this software package:

```bibtex
@software{marimba,
  author  = {Amadori, Solano},
  title   = {Marimba: typed, calibrated decisions about short and streaming audio},
  year    = {2026},
  url     = {https://github.com/Charcot-AI/marimba-api},
  version = {0.1.2}
}
```

## License

Code and the converted head and readout weights are Apache-2.0. The frozen
Qwen3-ASR-0.6B encoder is Apache-2.0; vendored code is attributed in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). The turn-end readout was trained on
`pipecat-ai/smart-turn-data-v3.2-train` (CC BY 4.0); no dataset audio ships here.

An open-source project by [Charcot](https://charcot.ai).
