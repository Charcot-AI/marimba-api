# Batch and Hugging Face datasets

```python
rows = model.batch(
    [{"path": "first.wav"}, {"path": "second.wav"}],
    questions,
    batch_size=16,
)
```

The outer result follows input order; each inner list follows question order.
By default batch inference runs sequentially, one encoder forward per window, and
`batch_size` has no effect.

## Batched encoder forward

`batched=True` shares one encoder forward across up to `batch_size` eight-second
windows. It applies to encoder questions (`voice_activity`, `turn_end`, `same_speaker`,
`speaker_change`, `speaker_count`); a call that includes a decoder question keeps the
sequential path.

```python
rows = model.batch(audios, questions, batch_size=16, batched=True)
rows = model.decide_windows(windows, questions, batched=True)
decisions = model.decide_pairs(Question.score("Is this the same speaker?"), pairs, batched=True)
```

`decide_windows` takes caller-assembled eight-second windows and `decide_pairs` builds
them from `(reference, segment)` audio with `marimba.assemble_pair`: the last 4 s of the
reference and the first 4 s of the segment for `same_speaker`, the last 6 s and the first
2 s for `speaker_change`. A part shorter than its slot is repeated to fill it.

On a GPU the batched path stays on the device: the log-mel spectrogram is computed
there for the whole batch, the forward stops at the last layer the questions read
(`voice_activity` 9, the speaker tasks 12, `turn_end` 18), captured layers are copied to
the host only when a readout needs them, and windows that share a question and a tail go
through the speech head in one pass. Measured on one NVIDIA L4 in bfloat16 with consecutive
windows of a public LibriSpeech sample, batch size 16, one thread: 2.7 ms per window for
the three questions of a streaming hop (`voice_activity`, `speaker_change`,
`speaker_count`), 2.2 ms for a `same_speaker` pair and 2.5 ms for `turn_end`.

The sequential path stays the default because the two paths are not bit-identical: the
batched forward runs attention over a stacked tensor, so results differ in the last
float bits. `scripts/check_batched_encoder.py` measures the difference on consecutive
eight-second windows of a public LibriSpeech sample and fails when a probability moves
by more than `--tolerance` (default 0.01) or a default-threshold decision flips:

```sh
python scripts/check_batched_encoder.py --device cpu --windows 64 --batch-size 16
```

Measured on CPU in float32, 60 windows, batch size 16:

| Question | Largest probability change | Decision flips |
|---|---:|---:|
| `turn_end` | 0.000015 | 0 |
| `voice_activity` (2 s) | 0.00000002 | 0 |
| `voice_activity` (1 s, threshold 0.1) | 0.000027 | 0 |
| `same_speaker` | 0.000005 | 0 |
| `speaker_change` | 0.0000002 | 0 |
| `speaker_count` | 0.000013 | 0 |

Measured on one NVIDIA L4, 60 windows, largest probability change at batch sizes 16
and 32; no decision flipped in any of the four runs:

| Question | float32, batch 16 | float32, batch 32 | bf16, batch 16 | bf16, batch 32 |
|---|---:|---:|---:|---:|
| `turn_end` | 0.00038 | 0.00038 | 0.0136 | 0.0120 |
| `voice_activity` (2 s) | 0.0000006 | 0.0000006 | 0.00003 | 0.00002 |
| `voice_activity` (1 s, threshold 0.1) | 0.0011 | 0.0011 | 0.0169 | 0.0148 |
| `same_speaker` | 0.00041 | 0.00041 | 0.0035 | 0.0050 |
| `speaker_change` | 0.00009 | 0.00009 | 0.0015 | 0.0016 |
| `speaker_count` | 0.00025 | 0.00025 | 0.0073 | 0.0035 |

Reduced precision moves probabilities more than float32 does: in bf16 two questions
pass the default tolerance of 0.01, so the script exits nonzero there even though no
decision flips. Run it on the deployment device and dtype before enabling
`batched=True` or `marimba serve --batched-encoder`, and keep thresholds away from
values that a change of this size would cross.
On an eight-thread CPU the batched forward took about 25% less time per window than
the sequential path at batch size 16; it is a throughput option, not a latency one.

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
