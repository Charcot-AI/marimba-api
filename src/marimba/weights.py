"""Pinned model manifest loading with per-file integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ENCODER_REPO = "Qwen/Qwen3-ASR-0.6B"
ENCODER_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DEFAULT_MODEL_REPO = "charcot-ai/marimba"
DEFAULT_MODEL_REVISION = "ad374b4e61202c8425d1b3b2766bef985b01df45"
AUDIO_TOWER_PREFIX = "thinker.audio_tower."
_ENCODER_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")


def load_manifest(repo: str, revision: str | None) -> tuple[dict[str, Any], dict[str, Path]]:
    resolved = revision or (DEFAULT_MODEL_REVISION if repo == DEFAULT_MODEL_REPO else None)
    if resolved is None or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        resolved = HfApi().model_info(repo, revision=resolved).sha
    if resolved is None:
        raise ValueError("Could not resolve an immutable model revision")
    manifest = json.loads(Path(hf_hub_download(repo, "config.json", revision=resolved)).read_text())
    if manifest.get("format_version") != 1 or manifest.get("encoder") != {
        "repo": ENCODER_REPO,
        "revision": ENCODER_REVISION,
    }:
        raise ValueError("Unsupported manifest format or encoder revision")
    paths = {}
    for lane in manifest["lanes"].values():
        for name, metadata in lane.get("files", {}).items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe model artifact path")
            path = Path(hf_hub_download(repo, name, revision=resolved))
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"SHA256 mismatch for model artifact {name}")
            paths[name] = path
    manifest["resolved_revision"] = resolved
    return manifest, paths


def audio_tower_tensors(checkpoint: Path) -> dict[str, Any]:
    """The audio-tower tensors of a full encoder checkpoint, read without the rest of the file."""
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt") as source:
        tensors = {
            name[len(AUDIO_TOWER_PREFIX) :]: source.get_tensor(name)
            for name in source.keys()  # noqa: SIM118
            if name.startswith(AUDIO_TOWER_PREFIX)
        }
    if not tensors:
        raise ValueError("The encoder checkpoint has no audio-tower tensors")
    return tensors


def load_audio_encoder(directory: str | Path | None) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Audio-tower configuration, tensors and the directory of the feature-extractor config.

    Without ``directory`` the pinned encoder checkpoint is fetched and only its audio tower
    is read; with one, the files written by :func:`export_audio_encoder` are read instead.
    """
    if directory is None:
        snapshot = Path(
            snapshot_download(
                ENCODER_REPO, revision=ENCODER_REVISION, allow_patterns=list(_ENCODER_FILES)
            )
        )
        config = json.loads((snapshot / "config.json").read_text())
        audio_config = config["thinker_config"]["audio_config"]
        return audio_config, audio_tower_tensors(snapshot / "model.safetensors"), snapshot
    from safetensors.torch import load_file

    path = Path(directory)
    config = json.loads((path / "config.json").read_text())
    if config.get("source") != {"repo": ENCODER_REPO, "revision": ENCODER_REVISION}:
        raise ValueError("The exported audio encoder does not match the pinned encoder revision")
    return config["audio_config"], load_file(path / "model.safetensors"), path


def export_audio_encoder(directory: str | Path) -> Path:
    """Write the audio tower of the pinned encoder checkpoint, and nothing else of it, to
    ``directory``: what the encoder lanes read, about a fifth of the checkpoint."""
    from safetensors.torch import save_file

    audio_config, tensors, snapshot = load_audio_encoder(None)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": {"repo": ENCODER_REPO, "revision": ENCODER_REVISION},
                "audio_config": audio_config,
            },
            indent=2,
        )
    )
    (path / "preprocessor_config.json").write_bytes(
        (snapshot / "preprocessor_config.json").read_bytes()
    )
    return path
