---
license: apache-2.0
language:
- en
tags:
- audio
- voice-activity-detection
- turn-taking
library_name: marimba
model-index:
- name: Marimba
  results: []
---

# Marimba

**Author: Solano Amadori (charcot.ai)**

Pre-release question-driven audio decisions using a frozen Qwen3-ASR-0.6B backbone,
a turn-end readout and a question-conditioned head. This is not a general-purpose
audio understanding or speaker-verification system.

## License and release status

- Package code: Apache-2.0, as recorded in the repository LICENSE.
- Converted readout and head weights: **Apache-2.0**.
- **[PENDING] public weight distribution and release artifact location**.
  This scaffold does not announce a public checkpoint or hosted service.

## Third-party attribution

- **Qwen/Qwen3-ASR-0.6B**, by the Qwen team: Apache-2.0. The pinned
  [model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md)
  declares `license: apache-2.0` at revision
  `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. See THIRD_PARTY_NOTICES.md
  for attribution and licensing of the vendored Qwen3-ASR modeling code.
- **Smart Turn readout training data: `pipecat-ai/smart-turn-data-v3.2-train`**,
  by Pipecat AI and the dataset card's credited contributors: **CC BY 4.0**.
  The readout training configuration pins revision
  `f8c95d284595defb6b88a715c356cbc5c31b7ad1`. Its
  [dataset card](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train/blob/f8c95d284595defb6b88a715c356cbc5c31b7ad1/README.md)
  declares `license: cc-by-4.0`; the License section confirms Creative Commons
  Attribution 4.0 International and the card credits the audio contributors.
  Dataset licensing is separate from the converted weights' license; no dataset
  audio is distributed with this package.
- **Voice-activity head training data:** attribution [PENDING].
  Public pinned training source: `pipecat-ai/smart-turn-data-v3.2-train`, revision
  `f8c95d284595defb6b88a715c356cbc5c31b7ad1`.

## Intended use

Research and application prototyping on short audio windows, with human review
where decisions matter. Voice activity is validated with Portuguese prompts and held-out
Arabic, Hindi, Korean and Russian prompts; recognized questions use the canonical prompt.
Not intended for diagnosis, surveillance, identity verification or safety-critical
automation. Obtain consent and appropriate rights for submitted audio.

## Installed capabilities

| Task | Kind | Lane | Status | Evidence / limitation |
|---|---|---|---|---|
| turn_end | score | encoder/readout | validated | Public AUROC 0.930; domain shift matters |
| voice_activity | noul | encoder/head | validated | validated with Portuguese prompts and held-out Arabic, Hindi, Korean and Russian prompts; recognized questions use the canonical prompt; AMI 0.995 and VoxConverse 0.9996 AUROC |
| language | choice | decoder | validated | Native 30-language LID; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| term | noul | decoder | experimental | Raw exact-match fraction of greedy + four beam hypotheses; uncalibrated |
| pii | noul | decoder | experimental with a supplied detector; otherwise unavailable | Caller-supplied detector on greedy transcript; no detector included |
| same_speaker | noul / score | encoder/readout | validated | AMI AUROC 0.998, VoxConverse 0.996; both 4 s halves single-speaker; on par with ECAPA |
| speaker_change | noul / score | encoder/readout | validated | dominant speaker in last 2 s vs previous 6 s; AMI AUROC 0.966, VoxConverse 0.980 |
| speaker_count | choice | encoder/readout | validated | 0/1/2/3+ speakers; accuracy AMI 0.778, VoxConverse 0.859; undercounts 3+ |
| audio_class | choice | encoder/head | unavailable | No speech/music/noise claim |

Decoder context above eight seconds is **unvalidated**. The API accepts up to
30 seconds for decoder lanes, but context-handling smoke checks do not validate
accuracy. Head/readout windows remain eight seconds.

Diarization uses raw L12 frames from the shared encoder pass, stored at float16
precision before float32 pooling. All eight seconds must be valid: partial or
masked windows return `None` for value, probability and probabilities.
Speaker count requires choice options exactly `0`, `1`, `2`, `3+`, in any order.
Binary diarization supports `noul` and `score`; score values are calibrated
positive-class probabilities. No speaker identities are produced.

PII questions apply a caller-supplied detector to the greedy transcript; the package
does not include a detector. The synthetic pt-BR evaluation scored item F1 0.904;
the capability documentation gives the measured results and limitations.

## Evidence

Turn-end AUROC is **0.930 versus Smart Turn v3.2 0.982** on Smart Turn's evaluation,
which is **in-distribution for Smart Turn**. On one out-of-domain meeting corpus,
**0.716 versus 0.712 is a tie**. These are historical comparisons specified for
this release, not results from the executable quickstart notebooks. They should
not be read as universal rankings or paired-significance claims.

> on one out-of-domain meeting corpus, 200 labeled windows reduced turn-end ECE from 0.20 to 0.011

AUROC unchanged within ±0.01.

Per-domain calibration uses labeled binary windows, a frozen model and a
one-dimensional positive-slope Platt map. This does not train the model weights,
guarantee transfer to another domain, or calibrate choice questions.

## Measured latency

Eight-second windows, batch one; three warmups excluded; 250 encoder/head windows,
100 decoder windows. CPU: Intel Xeon family 6/model 94, AVX2, eight threads.
GPU: NVIDIA L4. Torch 2.8.0+cu128, Transformers 4.57.6, SDPA, head fp32.

| Component | CPU fp32 p50 / p95 ms | L4 fp32 p50 / p95 ms |
|---|---:|---:|
| Encoder | 197.85 / 691.33 | 13.92 / 14.81 |
| Readout + VAD | 4.92 / 19.57 | 3.19 / 3.61 |
| Shared decoder prefix | 438.29 / 1598.71 | 33.54 / 35.12 |
| LID (five ISO choices) | 408.82 / 1689.72 | 174.64 / 181.53 |
| Term n-best | 4845.15 / 10252.40 | 1442.24 / 2436.85 |

LID/term exclude the shared prefix; component percentiles are not additive.
Readout includes a raw-logit parity diagnostic. Part of the CPU baseline overlapped
development on a shared VM. Other scores all 30 languages and was not timed here.
No real-time guarantee is claimed. See [benchmarks](../docs/benchmarks.md) for Mac
CPU measurements; no MPS latency was measured.

Current torch CUDA auto dtype prefers bf16 when supported by torch, then fp16 on
native-FP16 devices, then fp32; CPU auto stays fp32. Auto does not select MPS.
Explicit MPS auto dtype remains float16 when free memory is unknown or below 8 GiB,
otherwise float32. MPS is not validated: torch 2.8.0 outputs did not match CPU in
our bias-linear test, a mismatch not seen with torch 2.7.1.
The NumPy readout retains its existing precision for parity.

Eligibility was absolute binary-lane AUROC change <=0.005 and unchanged LID accuracy
versus CPU fp32. CUDA fp16/bf16 and CPU/CUDA beam2 passed. ONNX export failed on unsupported
`aten::pad_sequence` at opset 17; ONNX logit difference and decision agreement
are N/A. Teacher-forced term scoring is only a compute estimate, not implemented.

Encoder INT8 and greedy-only term variants are opt-in, quality-reducing report-only script
options, not model-loading or bench CLI knobs: turn-end AUROC change -0.090 for
encoder INT8 and term AUROC change -0.010 for greedy-only failed the parity gate.

## Reproducibility and limitations

The package pins an exact revision of this repository.
The manifest verifies per-lane files with SHA256.
The latency corpus uses Smart Turn v3.2 train revision
`f8c95d284595defb6b88a715c356cbc5c31b7ad1` and FLEURS validation revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, not test splits.
See the package docs/benchmarks.md for protocol and the separately retained
latency_matrix.json / latency_matrix.md measurement artifacts.

Accent, language, noise, overlap, microphone and domain changes can affect decisions.
Closed-set LID percentages do not establish open-set detection: Other aggregates
the unlisted supported languages, not all languages in the world. Lexical hypothesis
fractions are not calibrated occurrence probabilities.
Unavailable capabilities return no prediction.

**[PENDING] archival citation/DOI and public links for the historical evaluation
and full measurement artifacts**. No citation or release URL is invented here.

## Usage

Install the package and torch extra from the source checkout, authenticate outside
the code if access is required, then:

```python
from marimba import Marimba, Question

model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
decisions = model("speech.wav", [Question.noul("Is someone speaking?")])
print([d.to_dict() for d in decisions])
```

See the source checkout's docs/ index and executed notebooks for audio, batch,
datasets.map, streaming and scheduling examples. Empty audio is rejected.
