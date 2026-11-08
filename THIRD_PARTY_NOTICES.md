# Third-party notices

## Qwen3-ASR 0.0.6

`src/marimba/_vendor/qwen_asr/` contains the Transformers configuration,
modeling, and processor modules from the `qwen-asr==0.0.6` distribution,
plus only the output-parsing/repetition-normalization helpers used at inference.
Source: https://github.com/QwenLM/Qwen3-ASR (Apache License 2.0).
Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
Original notices are retained in each file; the license is included in that directory.
No web UI, forced aligner, serving, vLLM, or dataset code is included.

The three core modules are copied without behavioral changes. The utility file
is a subset of the upstream inference utilities. Upstream SHA256 values:

```
configuration_qwen3_asr.py: acf6c3f1cb3dc1ea0df621a11f7df6ccca109be73b6bf51508b5052942f5bec1
modeling_qwen3_asr.py: 2fb5d98da1933748f5117ee05ce4e7150c9ead8154fb8e25f7af3968b853adc7
processing_qwen3_asr.py: a61f1fbfc06e3dfc63a2b8d15e3b29e98b032792c3fb7d0e01c04ca9c2181ffb
```
