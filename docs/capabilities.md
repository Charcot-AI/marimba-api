# Capability reference

Statuses describe the installed frozen checkpoint, not every possible question.
Voice activity is validated with Portuguese prompts and held-out Arabic, Hindi, Korean and
Russian prompts; recognized questions use the canonical prompt; custom wording passes
through unchanged but is not automatically validated.

| Task | Kind | Lane | Installed status | Evidence / limitation |
|---|---|---|---|---|
| turn_end | score | encoder / readout | validated | AUROC 0.930 on Smart Turn v3.2 evaluation; 0.716 out of domain |
| voice_activity | noul | encoder / head | validated | validated with Portuguese prompts and held-out Arabic, Hindi, Korean and Russian prompts; recognized questions use the canonical prompt; AMI AUROC 0.995, VoxConverse 0.9996 |
| language | choice | decoder | validated | Native Qwen LID, 30 supported languages; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| term | noul | decoder | experimental | Greedy + four beam hypotheses; raw exact-match fraction, no fitted lexical Platt map |
| pii | noul | decoder | experimental with a supplied detector; otherwise unavailable | Caller-supplied text detector on greedy transcript; no detector included |
| same_speaker | noul / score | encoder / head | unavailable | Not a released output |
| speaker_change | noul / score | encoder / head | unavailable | Not a released output |
| speaker_count | choice | encoder / head | unavailable | Not a released output |
| audio_class | choice | encoder / head | unavailable | No installed lane; no speech/music/noise claim |

This is the installed-model table; generic capability metadata alone does not
prove weights are available. Inspect `model.capabilities` after loading.
The historical decoder study's calibrated lexical scores must not be substituted
for this package's uncalibrated raw fractions.

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

With an in-house Portuguese text PII model as the detector, on 200 synthetic pt-BR TTS clips (100 with one PII entity: name, CPF, phone, email or address; 100 without): item-level precision 0.966, recall 0.850, F1 0.904 (95% CI 0.856–0.945); with reference transcripts the same text model scores F1 1.000, so misses come from transcription (CPF recall 0.60, phone 0.70). Not evaluated on real or English speech.

Turn-end comparison: Marimba 0.930 versus Smart Turn v3.2 0.982 on Smart Turn's
own evaluation (in-distribution for Smart Turn); an out-of-domain meeting corpus
gives 0.716 versus 0.712, a tie. These are historical evidence, not new measurements
from the notebooks or latency corpus. See [calibration](calibration.md).
All decoder context above eight seconds is **unvalidated**.
