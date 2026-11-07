# Calibrate on your domain

Per-domain calibration adjusts probabilities, not the frozen encoder or task
weights. Use representative labeled windows and evaluate on separate held-out
windows from your domain. Calibration accepts binary `noul` and `score`
questions; `choice` questions are not supported.

> on one out-of-domain meeting corpus, 200 labeled windows reduced turn-end ECE from 0.20 to 0.011

AUROC unchanged within ±0.01.

This is one observed result, not a guarantee for another domain. For reproducible
comparison, retain the question, model revision, labels, and calibration file.

```python
from marimba import Marimba, Question

model = Marimba.from_pretrained(backend="torch", device="cpu")
question = Question.score("Has the speaker finished their turn?")

# application_windows contains audio paths or decoded audio mappings.
# application_labels contains matching bool or 0/1 labels.
calibration = model.calibrate(application_windows, application_labels, question)
model.save_calibrations("calibrations.json")

model.load_calibrations("calibrations.json")
decisions = model.batch(evaluation_windows, [question])
assert decisions[0][0].calibrated
model.clear_calibration(question)
```

No `datasets.map` is required: the audio input is the same sequence accepted by
`model.batch`, including a datasets audio column. Arrays must be in decoded
mappings carrying `sampling_rate`. `batch_size=16` controls iteration grouping;
inference is currently sequential.

`method="platt"` fits a one-dimensional logistic map on `logit(probability)`
using NumPy Newton steps with weak regularization (large C), without sklearn.
The slope is constrained positive to preserve ranking; probability endpoints
are clipped for numerical stability. Both classes must be present. Fewer than
100 windows emits a warning. Re-fitting uses raw model probabilities, not the
previously calibrated values.

The frozen `Calibration` contains `question`, `kind`, `method`, `a`, `b`, `n`,
`n_positive`, and `created_with` (the immutable model revision, or `None` for a
synthetic model). `to_dict()` / `from_dict()` support JSON serialization.
Loading a file merges its entries into the model; it does not change weights or
verify transferability to a different revision. Calibration is keyed by exact
question text and kind, so changing the wording requires its own calibration.

Subsequent batch, stream, and direct-lane decisions use the attached mapping and
set `Decision.calibrated=True`. `probability` and `value` are adjusted; the raw
lane `score` and capability status remain unchanged. Unavailable results stay
unavailable. With no attached calibration, behavior is unchanged.
