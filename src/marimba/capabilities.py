"""Published task capability metadata; availability still requires model weights."""

from dataclasses import dataclass
from typing import Literal

from .questions import Kind


@dataclass(frozen=True)
class Capability:
    task: str
    status: Literal["validated", "experimental", "unavailable"]
    lane: str
    kinds: tuple[Kind, ...]
    evaluation: str


DEFAULT_CAPABILITIES = {
    item.task: item
    for item in (
        Capability("turn_end", "validated", "readout", ("score",), "Public turn-end AUROC 0.930"),
        Capability(
            "voice_activity",
            "validated",
            "head",
            ("noul",),
            "English prompts; AMI AUROC 0.995, VoxConverse 0.9996",
        ),
        Capability(
            "language",
            "validated",
            "decoder",
            ("choice",),
            "30 languages; FLEURS 100%, Common Voice 93% closed set",
        ),
        Capability("term", "validated", "decoder", ("noul",), "Torch only; public AUROC 0.92–0.97"),
        Capability(
            "same_speaker",
            "experimental",
            "head",
            ("noul", "score"),
            "Public performance near chance",
        ),
        Capability(
            "speaker_change",
            "experimental",
            "head",
            ("noul", "score"),
            "Public performance near chance",
        ),
        Capability("speaker_count", "experimental", "head", ("choice",), "Not publicly validated"),
        Capability("audio_class", "unavailable", "head", ("choice",), "Not evaluated"),
    )
}
