# Capability reference

Statuses describe the installed frozen checkpoint, not every possible question.
Voice activity is validated with Portuguese prompts and held-out Arabic, Hindi, Korean and
Russian prompts; recognized questions use the canonical prompt; custom wording passes
through unchanged but is not automatically validated.

Voice activity also accepts `seconds=2.0`, scoring only the final two seconds of
the eight-second encoder window with the same encoder pass. On held-out validation,
the tail temperature 0.7869027171235091 yielded AUROC 0.998, F1 0.993 and
specificity 0.967 at the default 0.225 threshold. This is the only supported tail;
other non-`None` durations and other tasks are rejected.

| Task | Kind | Lane | Installed status | Evidence / limitation |
|---|---|---|---|---|
| turn_end | score, noul | encoder / readout | validated | AUROC 0.930 on Smart Turn v3.2 evaluation; 0.716 out of domain |
| voice_activity | noul | encoder / head | validated | validated with Portuguese prompts and held-out Arabic, Hindi, Korean and Russian prompts; recognized questions use the canonical prompt; probabilities are temperature-scaled on validation; default threshold 0.225 from validation; AMI AUROC 0.995, VoxConverse 0.9996 |
| language | choice | decoder | validated | Native Qwen LID, 30 supported languages; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| term | noul | decoder | validated | Greedy + four beam hypotheses; raw exact-match fraction, no fitted lexical Platt map |
| pii | noul | decoder | experimental with a supplied detector; otherwise unavailable | Caller-supplied text detector on greedy transcript; no detector included |
| same_speaker | noul / score | encoder / readout | validated | AMI AUROC 0.998, VoxConverse 0.996; both 4 s halves single-speaker; on par with ECAPA |
| speaker_change | noul / score | encoder / readout | validated | dominant speaker in last 2 s vs previous 6 s; AMI AUROC 0.966, VoxConverse 0.980 |
| speaker_count | choice | encoder / readout | validated | 0/1/2/3+ speakers; accuracy AMI 0.778, VoxConverse 0.859; undercounts 3+ |
| audio_class | choice | encoder / head | unavailable | No installed lane; no speech/music/noise claim |

This is the installed-model table; generic capability metadata alone does not
prove weights are available. Inspect `model.capabilities` after loading.
The historical decoder study's calibrated lexical scores must not be substituted
for this package's uncalibrated raw fractions.

Turn-end `score` returns the Platt probability. `noul` compares that probability
with the question's threshold, defaulting to 0.5. No event threshold has been
fitted; the AUROC above does not establish performance at that default threshold.

## Diarization questions

All three diarization lanes use raw L12 frames from the same encoder pass as
turn-end and voice activity, quantized to float16 before float32 pooling.
They require a full eight-second window. If any sample is masked, value,
probability and probabilities are `None`; streaming continues normally.

```python
questions = [
    Question.noul("Is this the same speaker?"),
    Question.score("Did the speaker change?"),
    Question.choice("How many speakers?", ["0", "1", "2", "3+"]),
]
```

Speaker-count routing also accepts `How many people?` and `How many people are
talking?`. Term routing accepts quoted terms in `Was "marimba" spoken?` and
`Was 'hello' said?`; the quotation marks are required.

Same-speaker and change accept `noul` or `score`. Scores are calibrated
positive-class probabilities; `noul` compares that probability to the threshold.
Count requires exactly those four distinct string options, in any order; its
probability dictionary preserves the supplied keys and its value is the argmax.
These lanes do not identify speakers or produce speaker timelines.

Optional local readout parity fixtures can be checked with
`MARIMBA_DIARIZATION_REFERENCE=/path/to/reference uv run pytest -m real_model tests/test_diarization_reference.py`.
The directory contains each task's safetensors file and
`<dataset>_<task>.npz` files for `ami` and `voxconverse`, with `frames` and
`probabilities` arrays covering at least 50 windows each. Without the variable,
the test skips; these fixtures are not distributed with the package.

## PII questions

PII questions apply a caller-supplied text detector to the greedy transcript. The package
does not ship a detector; without `pii_detector`, the capability is unavailable.

```python
from marimba import Marimba, Question


def contains_personal_information(transcript: str) -> bool:
    return "@" in transcript


model = Marimba.from_pretrained(pii_detector=contains_personal_information)
question = Question.noul("Does this audio contain any personal information?")
decision = model("speech.wav", [question])[0]
```

The detector example is illustrative only; choose a detector appropriate for your use.

With a private, curated PII pipeline as the detector, on 200 synthetic pt-BR TTS clips (100 with one PII entity: name, CPF, phone, email or address; 100 without): item-level precision 0.966, recall 0.850, F1 0.904 (95% CI 0.856–0.945); with reference transcripts the same pipeline scores F1 1.000, so misses come from transcription (CPF recall 0.60, phone 0.70). End to end on the audio, the same pipeline with its own production transcriber scores F1 0.975 (95% CI 0.949–0.995). Not evaluated on real or English speech.

Turn-end comparison: Marimba 0.930 versus Smart Turn v3.2 0.982 on Smart Turn's
own evaluation (in-distribution for Smart Turn); an out-of-domain meeting corpus
gives 0.716 versus 0.712, a tie. These are historical evidence, not new measurements
from the notebooks or latency corpus. See [calibration](calibration.md).
All decoder context above eight seconds is **unvalidated**.
