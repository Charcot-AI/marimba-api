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
        Capability(
            "turn_end", "validated", "readout", ("score", "noul"), "Public turn-end AUROC 0.930"
        ),
        Capability(
            "voice_activity",
            "validated",
            "head",
            ("noul",),
            "validated with Portuguese prompts and held-out Arabic, Hindi, Korean and Russian "
            "prompts; recognized questions use the canonical prompt; "
            "AMI AUROC 0.995, VoxConverse 0.9996",
        ),
        Capability(
            "language",
            "validated",
            "decoder",
            ("choice",),
            "30 languages; FLEURS 100%, Common Voice 93% closed set",
        ),
        Capability("term", "validated", "decoder", ("noul",), "Torch only; public AUROC 0.92–0.97"),
        Capability("pii", "unavailable", "decoder", ("noul",), "Requires a pii_detector"),
        Capability(
            "same_speaker",
            "validated",
            "readout",
            ("noul", "score"),
            "AMI AUROC 0.998, VoxConverse 0.996; both 4 s halves single-speaker; on par with ECAPA",
        ),
        Capability(
            "speaker_change",
            "validated",
            "readout",
            ("noul", "score"),
            "dominant speaker in last 2 s vs previous 6 s; AMI AUROC 0.966, VoxConverse 0.980",
        ),
        Capability(
            "speaker_count",
            "validated",
            "readout",
            ("choice",),
            "0/1/2/3+ speakers; accuracy AMI 0.778, VoxConverse 0.859; undercounts 3+",
        ),
        Capability("audio_class", "unavailable", "head", ("choice",), "Not evaluated"),
    )
}
