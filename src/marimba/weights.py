"""Pinned model manifest loading with per-file integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ENCODER_REPO = "Qwen/Qwen3-ASR-0.6B"
ENCODER_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DEFAULT_MODEL_REPO = "charcot-ai/marimba"
DEFAULT_MODEL_REVISION = "ad374b4e61202c8425d1b3b2766bef985b01df45"


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
