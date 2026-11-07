"""Question-driven audio inference interfaces."""

from .capabilities import Capability
from .decision import Decision, StreamUpdate
from .model import Marimba
from .questions import Question, UnsupportedQuestion
from .runtime import RuntimeInfo
from .stream import Stream
from .windows import MAX_DECODER_SECONDS, EncodedWindow, WindowSnapshot

__all__ = [
    "Capability",
    "Decision",
    "EncodedWindow",
    "MAX_DECODER_SECONDS",
    "Marimba",
    "Question",
    "RuntimeInfo",
    "Stream",
    "StreamUpdate",
    "UnsupportedQuestion",
    "WindowSnapshot",
]
__version__ = "0.1.0"
