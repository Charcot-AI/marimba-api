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

The package's beam/n-best defaults are unchanged: greedy plus four beam hypotheses.
Encoder INT8 and greedy-only term generation are **opt-in, quality-reducing**
variants exposed only by the report-only `scripts/latency_matrix.py --variant`
interface, not public model-loading or bench CLI options.

Current torch CUDA auto dtype prefers bf16 when supported, then fp16 on native-FP16
devices, then fp32; CPU auto is fp32.

ONNX 1.23.2 / ORT 1.31.1: legacy export (opset 17) failed on unsupported
`aten::pad_sequence`. ORT INT8 was blocked; maximum absolute logit difference and
decision agreement are **N/A**, not zero. CPU BF16 was not measured without native
BF16 support. The encoder-only thread sweep selected eight; tuned reuses baseline.

LID and term exclude the shared prefix, which is added once. Component percentile
sums are not end-to-end percentiles. Readout includes a raw-logit parity diagnostic.
Part of the CPU baseline overlapped development on a shared VM.
The matrix uses five ISO language choices, not the new 30-language Other path.
No real-time guarantee or >8-second decoder-quality claim is made.

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

## End-to-end per-path latency, L4 bf16

Full public-API calls on one NVIDIA L4, batch one, eight-second windows,
`dtype="auto"` asserted to resolve to bf16. Package code is frozen at
`a38383ecc73c6192dbffaf9950289189929600df`; model and Qwen revisions are recorded in
the [JSON report](../benchmarks/l4_e2e_latency.json), alongside resolved RuntimeInfo,
versions, raw call latencies and job duration/cost.

Milliseconds; the final column is the bootstrap 95% confidence interval for p95.

| Path | n | p50 | p95 | p95 95% CI |
|---|---:|---:|---:|---:|
| Turn-end (L18 readout) | 300 | 12.23 | 13.31 | 13.04–14.08 |
| Voice activity (question-conditioned head) | 300 | 14.40 | 15.77 | 15.47–15.97 |
| Language (native LID) | 300 | 153.67 | 160.80 | 159.23–163.55 |
| Term (default n-best) | 200 | 1399.44 | 2349.14 | 2136.89–2443.73 |

Twenty warmup calls per path were discarded. Timed calls were interleaved
turn-end → voice activity → language → term, omitting term after call 200 and
continuing the other paths to 300. Each timer wraps exactly one
`model(audio, [question])`, with CUDA synchronization before starting and before
stopping, after the Decision list returns. This includes input normalization,
encoder and lane work; there is no cross-question sharing. Model loading and
corpus loading are excluded.

The existing left-padded, 16 kHz float32 matrix windows were preloaded as public
audio mappings. The public API constructs its own mask from each full eight-second
input; the matrix's original masks are retained only as provenance. Per-lane clips
are unchanged: 100 Smart Turn train for turn-end, 50 FLEURS validation plus 50
synthetic negatives for voice activity, and 100 FLEURS validation for each decoder
path. Clips cycle in their matrix order; measurement continues at clip 20 after
warmup. [Clip IDs, pinned revisions and window hashes](../benchmarks/l4_e2e_clip_ids.json)
are committed, with the file's SHA256 in the report. No audio or transcripts are
distributed here.

Language options are `en, pt, es, fr, de`. The fixed term is `"marimba"`;
greedy plus four-beam n-best defaults are unchanged. Percentiles use NumPy's linear
method. The p95 intervals are percentile bootstrap intervals over calls: 1,000
resamples, `numpy.random.default_rng(42)` reset per path, and 2.5th/97.5th
percentiles of resampled p95 values. They are not intervals over independent clips.

The reusable runner is [scripts/l4_e2e_latency.py](../scripts/l4_e2e_latency.py).
With the original corpus files and the frozen package installed on an L4:

```sh
python scripts/l4_e2e_latency.py \
  --corpus matrix_corpus.npz --records matrix_corpus.json \
  --corpus-sha256 7cda39dde5231a59701036a3d71f9931ee23a9dd47ade4674fce01d62d000408 \
  --code-commit a38383ecc73c6192dbffaf9950289189929600df \
  --package-sha256 "$(sha256sum package.tar.gz | cut -d ' ' -f 1)" \
  --output l4_e2e_latency.json --clip-ids l4_e2e_clip_ids.json
```

## End-to-end per-path latency, Mac CPU

Same protocol as L4: public `model(audio, [question])` calls, batch one, eight-second
windows, 20 warmup calls discarded, and 300 timed calls per path, interleaved.
The p95 95% bootstrap confidence intervals use 1,000 replicates with seed 42.
Measurements used CPU fp32 on a virtualized Apple M4 Pro (12 cores, 16 GB), macOS
26.6.2, with torch 2.8.0. Units are milliseconds.

| Path | Threads | p50 | p95 | p95 95% CI |
|---|---:|---:|---:|---:|
| Turn-end | 4 | 105.0 | 109.4 | 108.8–110.0 |
| Voice activity | 4 | 108.0 | 112.2 | 111.7–112.6 |
| Language | 4 | 1264.9 | 1297.6 | 1285.6–1305.2 |
| Turn-end | 8 | 99.5 | 111.2 | 106.5–125.7 |
| Voice activity | 8 | 103.1 | 117.3 | 111.6–122.3 |
| Language | 8 | 1437.4 | 1555.0 | 1531.9–1610.4 |

Term was not measured on Mac CPU. No MPS latency was measured. These measurements
are not a real-time guarantee.
