"""Optional parity against local, externally supplied reference fixtures."""

import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from marimba._diarization import TASKS, DiarizationReadout, task_features


@pytest.mark.real_model
@pytest.mark.parametrize("dataset", ["ami", "voxconverse"])
@pytest.mark.parametrize("task", TASKS)
def test_local_diarization_reference(dataset, task):
    directory = os.environ.get("MARIMBA_DIARIZATION_REFERENCE")
    if not directory:
        pytest.skip("Set MARIMBA_DIARIZATION_REFERENCE to local reference fixtures")
    root = Path(directory)
    readout = DiarizationReadout(task, load_file(root / f"{task}.safetensors"))
    with np.load(root / f"{dataset}_{task}.npz", allow_pickle=False) as fixture:
        frames, expected = fixture["frames"], fixture["probabilities"]
    assert len(frames) >= 50
    features = np.stack(
        [task_features(task, f.astype(np.float16).astype(np.float32)) for f in frames]
    )
    actual = readout.predict(features)
    difference = float(np.max(np.abs(actual - expected)))
    print(f"{dataset}/{task}: n={len(frames)}, max_abs_probability_diff={difference:.12g}")
    assert difference <= 1e-4
