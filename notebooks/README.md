# Notebooks

## Colab quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Charcot-AI/marimba-api/blob/main/notebooks/colab_quickstart.ipynb)

The [Colab quickstart](colab_quickstart.ipynb) installs the published PyPI package and
demonstrates supported question types with public LibriSpeech speech, including a
transcript-checked term example and a two-speaker clip. It is an API example, not
an accuracy measurement.

## Executed CPU notebooks

1. [Quickstart](01_quickstart.ipynb)
2. [datasets.map](02_datasets_map.ipynb)
3. [Chunked streaming](03_streaming.ipynb)

All use real checkpoint inference on CPU, four Torch threads, two public FLEURS
validation rows, pinned at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
They retain small textual outputs, not audio, transcripts, credentials or file paths.
No task-quality or performance estimate is derived from these examples.
Decoder context above eight seconds is **unvalidated**.

Install from the checkout with `pip install '.[torch,datasets]'`, then install
`nbclient nbformat ipykernel` and a TorchCodec build compatible with Torch/FFmpeg.
Use an authenticated kernel when access is required; never paste tokens into cells.
The pinned model revision is the package default:
`bc980b4e6eb34e7fea96ea8b0884c2093a845dc5`.
Converted head/readout weights are Apache-2.0; see the model card.

To execute again, start Jupyter with that environment's kernel and run all cells
in order. For automated execution, use nbclient with that kernel. The datasets
notebook disables map-cache reuse so its output is recomputed.
