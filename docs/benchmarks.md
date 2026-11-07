# Measured benchmarks

Summary of the recorded eight-second, batch-one latency matrix. Three warmups
excluded; 250 encoder/head windows and 100 decoder windows. Units: milliseconds,
p50 / p95. CPU: Intel Xeon family 6/model 94, eight vCPUs, AVX2, eight Torch threads.
GPU: NVIDIA L4. Torch 2.8.0+cu128, Transformers 4.57.6, SDPA; heads stay float32.
Model revision: `bc980b4e6eb34e7fea96ea8b0884c2093a845dc5`.

| Variant | Encoder | Readout + VAD | Shared prefix | LID (five options) | Term | Eligible |
|---|---:|---:|---:|---:|---:|---|
| CPU fp32 | 197.85 / 691.33 | 4.92 / 19.57 | 438.29 / 1598.71 | 408.82 / 1689.72 | 4845.15 / 10252.40 | yes |
| CPU encoder INT8 | 131.83 / 359.40 | 4.23 / 18.44 | 367.99 / 812.77 | 334.48 / 856.43 | 3487.23 / 7885.45 | no |
| L4 fp32 | 13.92 / 14.81 | 3.19 / 3.61 | 33.54 / 35.12 | 174.64 / 181.53 | 1442.24 / 2436.85 | yes |
| L4 fp16 | 12.62 / 13.67 | 3.29 / 3.76 | 29.06 / 32.34 | 184.28 / 191.43 | 1355.26 / 2280.00 | yes |
| L4 bf16 | 12.16 / 12.97 | 3.29 / 3.70 | 28.61 / 30.18 | 182.72 / 192.43 | 1350.65 / 2275.29 | yes |
| L4 greedy | 14.15 / 15.14 | 3.22 / 3.62 | 33.54 / 35.58 | 172.20 / 177.24 | 675.51 / 1138.71 | no |

Eligibility requires each binary lane's absolute AUROC change <=0.005 and unchanged
LID accuracy versus CPU fp32. CPU/L4 beam2 passed at term change -0.0050; CPU/L4
greedy failed at -0.0100. Encoder INT8 failed at turn-end -0.0900 and term -0.0100.
This gate is not a production or calibrated-confidence recommendation.

ONNX 1.23.2 / ORT 1.31.1: legacy export (opset 17) failed on unsupported
`aten::pad_sequence`. ORT INT8 was blocked; maximum absolute logit difference and
decision agreement are **N/A**, not zero. CPU BF16 was not measured without native
BF16 support. The encoder-only thread sweep selected eight; tuned reuses baseline.

LID and term exclude the shared prefix, which is added once. Component percentile
sums are not end-to-end percentiles. Readout includes a raw-logit parity diagnostic.
Part of the CPU baseline overlapped development on a shared VM.
The matrix uses five ISO language choices, not the new 30-language Other path.
No real-time, Mac measurement or >8-second decoder-quality claim is made.

Provenance: Smart Turn v3.2 train revision
`f8c95d284595defb6b88a715c356cbc5c31b7ad1`; FLEURS validation revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. No test splits were used for this matrix.
The original `latency_matrix.json` and `latency_matrix.md` measurement artifacts
are maintained separately from the source checkout; no audio is distributed here.
Reusable measurement/export tooling is `scripts/latency_matrix.py`.

Teacher-forced term scoring is not implemented. A rough LID-suffix compute proxy
suggested 82–245 ms CPU or 35–105 ms L4 incremental compute, plus encoder/prefix if
uncached. This is an estimate, not measured term scoring: term-token likelihood is
not term-occurrence probability and needs its own evaluation and calibration.
