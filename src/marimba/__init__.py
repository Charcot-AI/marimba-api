"""Question-driven audio inference interfaces."""

from .calibration import Calibration
from .capabilities import Capability
from .decision import Decision, StreamUpdate
from .model import Marimba
from .pii import PIIDetector
from .questions import Question, UnsupportedQuestion
from .runtime import RuntimeInfo
from .stream import Stream
from .windows import (
    MAX_DECODER_SECONDS,
    PAIR_LAYOUTS,
    EncodedWindow,
    WindowSnapshot,
    assemble_pair,
)

__all__ = [
    "Calibration",
    "Capability",
    "Decision",
    "EncodedWindow",
    "MAX_DECODER_SECONDS",
    "PAIR_LAYOUTS",
    "Marimba",
    "PIIDetector",
    "Question",
    "RuntimeInfo",
    "Stream",
    "StreamUpdate",
    "UnsupportedQuestion",
    "WindowSnapshot",
    "assemble_pair",
]
__version__ = "0.1.2"
