# Batch and Hugging Face datasets

```python
rows = model.batch(
    [{"path": "first.wav"}, {"path": "second.wav"}],
    questions,
    batch_size=16,
)
```

The outer result follows input order; each inner list follows question order.
Batch inference currently runs sequentially. The batch_size argument is reserved
for backend batching; it is not a throughput or vectorization claim.

Install the datasets extra from a checkout: `pip install '.[torch,datasets]'`.
For Hugging Face datasets with a compatible audio decoder:

```python
from datasets import Audio

ds = ds.cast_column("audio", Audio(sampling_rate=16000))
scored = ds.map(model.map_fn(questions), batched=True, batch_size=2)
```

This adds structured columns `marimba_0`, `marimba_1`, etc.; original columns remain.
Each contains a JSON-safe Decision dictionary. Do not use multiprocessing with a
shared loaded model. A datasets map cache can contain predictions: disable reuse
with `load_from_cache_file=False` when deliberately recomputing results.

Recent datasets versions use TorchCodec for lazy decoding. Install a TorchCodec
version compatible with your Torch and FFmpeg versions if needed. Alternatively,
cast with `Audio(sampling_rate=16000, decode=False)`: datasets then returns encoded
bytes/path mappings, and Marimba itself decodes and resamples them. The cast alone
does not resample bytes when decode=False.

See the [executed datasets notebook](../notebooks/02_datasets_map.ipynb), which
loads two public FLEURS validation rows at an immutable revision. No test data or
audio payload is embedded in the notebook.
