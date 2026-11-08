"""Decode audio once, and assemble right-aligned inference windows."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
import soxr

AudioArray = npt.NDArray[np.float32]
Mask = npt.NDArray[np.bool_]
SAMPLE_RATE = 16000


def normalize_array(array: Any, *, channels_first: bool = False) -> AudioArray:
    data = np.asarray(array)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Audio samples must be int16 or floating point")
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=0 if channels_first else 1)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Audio must be finite mono or two-dimensional multichannel samples")
    return np.asarray(data, dtype=np.float32)


def decode_audio(audio: Any, sampling_rate: int | None = None) -> AudioArray:
    channels_first = False
    if hasattr(audio, "get_all_samples"):
        # Access lazy decoders only here, rather than materializing their mapping keys.
        samples = audio.get_all_samples()
        audio, sampling_rate = samples.data.cpu().numpy(), samples.sample_rate
        channels_first = True
    elif isinstance(audio, dict):
        if "array" in audio:
            audio, sampling_rate = audio["array"], audio["sampling_rate"]
        elif audio.get("bytes") is not None:
            audio, sampling_rate = sf.read(BytesIO(audio["bytes"]), dtype="float32")
        elif audio.get("path") is not None:
            audio, sampling_rate = sf.read(audio["path"], dtype="float32")
        else:
            raise ValueError("Audio mapping needs array, bytes, or path")
    elif isinstance(audio, (str, Path, bytes, bytearray)):
        source = BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        audio, sampling_rate = sf.read(source, dtype="float32")
    if sampling_rate is None or sampling_rate <= 0:
        raise ValueError("A positive sampling_rate is required for array audio")
    data = normalize_array(audio, channels_first=channels_first)
    if sampling_rate != SAMPLE_RATE and data.size:
        data = soxr.resample(data, sampling_rate, SAMPLE_RATE)
    return np.asarray(data, dtype=np.float32)


def right_aligned_window(audio: AudioArray, size: int = 128000) -> tuple[AudioArray, Mask]:
    if size <= 0:
        raise ValueError("Window size must be positive")
    window = np.zeros(size, dtype=np.float32)
    mask = np.zeros(size, dtype=np.bool_)
    count = min(size, len(audio))
    if count:
        window[-count:] = audio[-count:]
        mask[-count:] = True
    return window, mask
