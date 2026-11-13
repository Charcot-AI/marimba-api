# Marimba documentation

Marimba returns typed decisions over eight-second audio windows using frozen model
weights. Start with the [README](../README.md) for install and a first run.

| Guide | Covers |
|---|---|
| [Quickstart](quickstart.md) | First real-model decisions and the `Decision` fields |
| [Audio inputs](audio.md) | Files, bytes, arrays, lazy decoders, resampling and windows |
| [Batch and Hugging Face datasets](batch.md) | `model.batch` and `datasets.map` with `map_fn` |
| [Streaming](streaming.md) | `push`/`flush`, `ingest`/`decide`, Pipecat and LiveKit adapters |
| [Lane scheduling](scheduling.md) | Snapshots, encoded windows, split locks, decoder cadence, latency fields |
| [Runtime and backends](runtime.md) | `marimba doctor`, device and dtype selection, ONNX status |
| [Capabilities and evidence](capabilities.md) | Every task, its status, evidence and limits |
| [Measured benchmarks](benchmarks.md) | CPU, L4 and Mac latency with protocol |
| [Calibrate on your domain](calibration.md) | Per-question Platt maps from your own labels |
| [OpenAI-compatible server](openai-server.md) | `marimba serve` request, response and token accounting |
| [Hosting the server](hosting.md) | Image, Runpod options and cost estimate |

Also in this repository:

- [Executed notebooks](../notebooks/README.md): CPU inference on pinned public validation audio.
- [Examples](../examples): Pipecat, LiveKit and Vapi.
- [Gradio Space](../space/README.md): typed questions on synthetic example clips.
- [Model card](https://huggingface.co/charcot-ai/marimba) on the Hugging Face Hub.

No real-time service guarantee is provided.
