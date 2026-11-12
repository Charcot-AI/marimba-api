"""Deploy a source-bundled Gradio Space without changing public visibility."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = Path(__file__).resolve().parents[1]


def deployment_files() -> dict[str, Path | bytes]:
    files: dict[str, Path | bytes] = {
        "README.md": ROOT / "space" / "README.md",
        "requirements.txt": (ROOT / "space" / "requirements.txt").read_bytes(),
    }
    for name in ("pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        files[name] = ROOT / name
    for path in (ROOT / "src").rglob("*"):
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed"):
            files[path.relative_to(ROOT).as_posix()] = path
    for name in (
        "app.py",
        "requirements.txt",
        "generate_examples.py",
        "smoke_test.py",
        "deploy.py",
    ):
        files[f"space/{name}"] = ROOT / "space" / name
    examples = ROOT / "space" / "examples"
    clips = sorted(examples.glob("*.wav"))
    if len(clips) != 8:
        raise ValueError("Prepare all eight synthetic scene clips before deployment.")
    for path in [*clips, examples / "manifest.json"]:
        files[path.relative_to(ROOT).as_posix()] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="charcot-ai/marimba")
    parser.add_argument("--hardware", choices=["zero-a10g", "l4x1"], default="zero-a10g")
    args = parser.parse_args()
    files = deployment_files()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
        space_hardware=args.hardware,
        space_sleep_time=300 if args.hardware == "l4x1" else None,
    )
    if not api.space_info(args.repo_id).private:
        raise RuntimeError("Refusing to update a public Space.")
    api.request_space_hardware(
        args.repo_id,
        hardware=args.hardware,
        sleep_time=300 if args.hardware == "l4x1" else None,
    )
    api.add_space_secret(args.repo_id, key="HF_TOKEN", value=os.environ["HF_TOKEN"])
    obsolete_examples = [
        name
        for name in api.list_repo_files(args.repo_id, repo_type="space")
        if name.startswith("space/examples/") and name.endswith(".wav") and name not in files
    ]
    commit = api.create_commit(
        args.repo_id,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=source)
            for name, source in sorted(files.items())
        ]
        + [CommitOperationDelete(path_in_repo=name) for name in obsolete_examples],
        commit_message="feat: deploy typed multilingual audio decisions",
    )
    info = api.space_info(args.repo_id)
    if not info.private:
        raise RuntimeError("Space visibility verification failed.")
    print(f"Private Space: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Commit: {commit.oid}; requested hardware: {info.runtime.requested_hardware}")


if __name__ == "__main__":
    main()
