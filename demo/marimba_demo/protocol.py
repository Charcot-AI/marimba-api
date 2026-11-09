"""One in-flight hop bounds latency instead of building an inference queue."""

import struct
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16000
HOP_SAMPLES = 4000
HEADER = struct.Struct("<Id")


class Frame(NamedTuple):
    sequence: int
    captured_at_ms: float
    pcm: NDArray[np.int16]


def decode_frame(data: bytes) -> Frame:
    if len(data) != HEADER.size + HOP_SAMPLES * 2:
        raise ValueError("Expected a 12-byte header and exactly 4000 PCM16 samples.")
    sequence, captured_at_ms = HEADER.unpack_from(data)
    if not np.isfinite(captured_at_ms) or captured_at_ms < 0:
        raise ValueError("Invalid capture timestamp.")
    return Frame(
        sequence,
        captured_at_ms,
        np.frombuffer(data, dtype="<i2", offset=HEADER.size).copy(),
    )
