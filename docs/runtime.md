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

Auto dtype follows device/memory policy; it is not a fastest-dtype guarantee.
The measured L4 auto run selected float32. Explicit half-precision variants were
measured separately. Core imports do not load an inference engine. Model loading
and doctor runtime probing may import torch; only model loading fetches weights.
Core installation has no inference engine. The torch extra excludes Gradio;
the demo extra is opt-in and brings its web dependencies.

Mac/MPS is a discovery path, **not a measured result**. No Mac latency or MPS parity
is measured, and no real-time guarantee is claimed. See [benchmarks](benchmarks.md).
