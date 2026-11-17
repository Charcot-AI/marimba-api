# Quickstart

With Python 3.10 or newer:

```sh
pip install "marimba-audio[torch]"
marimba doctor
```

From a source checkout, use `pip install '.[torch]'` instead.

Authenticate with `hf auth login` or set `HF_TOKEN` if your configured model
repository requires access. Converted head/readout weights are Apache-2.0; see the
[model card](https://huggingface.co/charcot-ai/marimba). Do not put credentials in notebooks or source.

```python
from marimba import Marimba, Question

model = Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")
questions = [
    Question.score("Has the speaker finished?"),
    Question.noul("Is someone speaking?"),
]
decisions = model("speech.wav", questions)
print([decision.to_dict() for decision in decisions])
```

Files are decoded, mixed to mono, resampled to 16 kHz and right-aligned.
Only the last eight seconds are scored. Short audio is left-padded with a valid
mask; empty real-model audio is rejected. Silence with valid samples is supported.

`Decision` includes task, kind, status, value, probability, optional choice
probabilities, score, threshold, timestamps, latency and a calibrated flag.
A score value is numeric; a `noul` value is Boolean. An unavailable task returns
`None` values, never a simulated prediction. Latency excludes loading and decoding.

Language examples may use ISO codes or English names:
`Question.choice("What language?", ["English", "Portuguese", "Other"])`.
Other scores all 30 native languages and aggregates unlisted mass; it is not an
unknown-language detector. Duplicate aliases such as English plus en are rejected.

For integration without weights, use `marimba.testing.FakeMarimba`; its outputs
are synthetic and must not be presented as model predictions.
