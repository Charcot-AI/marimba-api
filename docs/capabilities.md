# Capability reference

Statuses describe the installed frozen checkpoint, not every possible question.
Use English prompts; task= explicitly routes custom wording but does not validate it.

| Task | Kind | Lane | Installed status | Evidence / limitation |
|---|---|---|---|---|
| turn_end | score | encoder / readout | validated | AUROC 0.930 on Smart Turn v3.2 evaluation; 0.716 out of domain |
| voice_activity | noul | encoder / head | validated | English prompts; AMI AUROC 0.995, VoxConverse 0.9996 |
| language | choice | decoder | validated | Native Qwen LID, 30 supported languages; closed-set FLEURS 100%, Common Voice 93% (300 each) |
| term | noul | decoder | experimental | Greedy + four beam hypotheses; raw exact-match fraction, no fitted lexical Platt map |
| same_speaker | noul / score | encoder / head | experimental | Public discrimination near chance; single-window score, not reference-pair verification |
| speaker_change | noul / score | encoder / head | unavailable | No installed lane |
| speaker_count | choice | encoder / head | unavailable | No installed lane |
| audio_class | choice | encoder / head | unavailable | No installed lane; no speech/music/noise claim |

This is the installed-model table; generic capability metadata alone does not
prove weights are available. Inspect `model.capabilities` after loading.
The historical decoder study's calibrated lexical scores must not be substituted
for this package's uncalibrated raw fractions.

Turn-end comparison: Marimba 0.930 versus Smart Turn v3.2 0.982 on Smart Turn's
own evaluation (in-distribution for Smart Turn); an out-of-domain meeting corpus
gives 0.716 versus 0.712, a tie. These are historical evidence, not new measurements
from the notebooks or latency corpus. See [calibration](calibration.md).
All decoder context above eight seconds is **unvalidated**.
