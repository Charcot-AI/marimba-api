---
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
- Upstream Qwen3-ASR modeling code and frozen backbone: Apache-2.0; upstream model
  license checked at revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
  See THIRD_PARTY_NOTICES.md for vendored-code attribution.
- Converted readout and head weights: **[PENDING] explicit redistribution license
  and provenance approval**. The checkpoint manifest and existing model-card
  metadata contain no weight-license declaration. Do not infer one from the
  package's code license. No blanket model-card license tag is asserted.
- **[PENDING] public weight distribution and release artifact location**.
  This scaffold does not announce a public checkpoint or hosted service.

## Intended use

Research and application prototyping on short audio windows, with human review
where decisions matter. English head prompts are the evaluated interface.
Not intended for diagnosis, surveillance, identity verification or safety-critical
automation. Obtain consent and appropriate rights for submitted audio.

## Installed capabilities

| Task | Kind | Lane | Status | Evidence / limitation |
|---|---|---|---|---|
| turn_end | score | encoder/readout | validated | Public AUROC 0.930; domain shift matters |
| voice_activity | noul | encoder/head | validated | English prompts; AMI 0.995 and VoxConverse 0.9996 AUROC |
| language | choice | decoder | validated | Native 30-language LID; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| term | noul | decoder | experimental | Raw exact-match fraction of greedy + four beam hypotheses; uncalibrated |
| same_speaker | noul / score | encoder/head | experimental | Near-chance public discrimination; not reference-pair verification |
| speaker_change | noul / score | encoder/head | unavailable | No installed lane |
| speaker_count | choice | encoder/head | unavailable | No installed lane |
| audio_class | choice | encoder/head | unavailable | No speech/music/noise claim |

Decoder context above eight seconds is **unvalidated**. The API accepts up to
30 seconds for decoder lanes, but context-handling smoke checks do not validate
accuracy. Head/readout windows remain eight seconds.

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
No real-time or measured Mac performance is claimed.

Current torch CUDA auto dtype prefers bf16 when supported by torch, then fp16 on
native-FP16 devices, then fp32; CPU auto stays fp32. MPS auto policy is unchanged.
The table explicitly measures fp32, not the new auto policy. The earlier L4 auto
benchmark used the old memory-based policy and selected fp32; its recorded results
must not be relabeled as bf16 measurements. The speech head stays fp32 and the NumPy
readout retains its existing precision for parity.

Eligibility was absolute binary-lane AUROC change <=0.005 and unchanged LID accuracy
versus CPU fp32. CUDA fp16/bf16 and CPU/CUDA beam2 passed. Encoder dynamic INT8 and
CPU/CUDA greedy were not eligible. ONNX export failed on unsupported
`aten::pad_sequence` at opset 17; ONNX logit difference and decision agreement
are N/A. Teacher-forced term scoring is only a compute estimate, not implemented.

Production beam/n-best defaults remain greedy plus four beam hypotheses. Encoder
INT8 and greedy-only term variants are opt-in, quality-reducing report-only script
options, not model-loading or bench CLI knobs: turn-end AUROC change -0.090 for
encoder INT8 and term AUROC change -0.010 for greedy-only failed the parity gate.

## Reproducibility and limitations

The package pins model revision `bc980b4e6eb34e7fea96ea8b0884c2093a845dc5`;
the manifest verifies per-lane files with SHA256. No weights changed for this card.
The latency corpus uses Smart Turn v3.2 train revision
`f8c95d284595defb6b88a715c356cbc5c31b7ad1` and FLEURS validation revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, not test splits.
See the package docs/benchmarks.md for protocol and the separately retained
latency_matrix.json / latency_matrix.md measurement artifacts.

Accent, language, noise, overlap, microphone and domain changes can affect decisions.
Closed-set LID percentages do not establish open-set detection: Other aggregates
the unlisted supported languages, not all languages in the world. Lexical hypothesis
fractions are not calibrated occurrence probabilities. Same-speaker is experimental.
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
