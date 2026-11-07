# Runtime and backends

```sh
marimba doctor
marimba bench --fake --warmup 3 --hops 20
```

doctor reports platform, installed extras, devices, available memory and ONNX
providers where detectable. Runtime discovery is not a successful model-load check.
The fake benchmark measures synthetic integration code, not real model inference.

Discovery prefers torch CUDA, torch MPS, ONNX providers (Core ML/CUDA/CPU), then
torch CPU. The current pretrained checkpoint lanes are **torch-only**: ONNX model
loading is not implemented, and the export spike failed. Installing the onnx extra
does not make these weights usable with ONNX Runtime.

For reproducible CPU execution use:
```python
model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
```

For torch CUDA, `dtype="auto"` selects bfloat16 when
`torch.cuda.is_bf16_supported()` is true, otherwise float16 on devices with native
FP16 arithmetic (compute capability 5.3 or newer), otherwise float32. CPU auto stays
float32. MPS is unchanged: float16 when free memory is unknown or below 8 GiB,
otherwise float32. Explicit dtype requests override auto. This is not a fastest-dtype
or available-memory guarantee. ONNX discovery behavior is unchanged; these checkpoint
lanes remain torch-only.

doctor reports `cuda_bf16_supported`, `cuda_fp16_supported` and the resolved selection
dtype. `model.runtime.dtype` and `RuntimeInfo.to_dict()["dtype"]` describe the resolved
backbone dtype, not head/readout precision. The speech head stays fp32 and the NumPy
readout keeps its existing precision.

The historical L4 auto benchmark selected float32 under the previous memory-based
policy. Its numbers are not a measurement of the new auto policy. Explicit fp16/bf16
variants were measured separately. Core imports do not load an inference engine. Model loading
and doctor runtime probing may import torch; only model loading fetches weights.
Core installation has no inference engine. The torch extra excludes Gradio;
the demo extra is opt-in and brings its web dependencies.

Production term defaults remain greedy plus four beam hypotheses. No encoder-INT8
or greedy-only switch is exposed by model loading or the bench CLI. Those variants
exist only in the opt-in report-only matrix script; both reduced quality and failed
the parity gate. See [benchmarks](benchmarks.md).

Mac/MPS is a discovery path, **not a measured result**. No Mac latency or MPS parity
is measured, and no real-time guarantee is claimed. See [benchmarks](benchmarks.md).
