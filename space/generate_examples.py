"""Cut reproducible examples from a licensed synthetic ElevenLabs scene assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

EXAMPLES = Path(__file__).parent / "examples"
SOURCE_SHA256 = "18516da0701d76fd1f21d5418ebc555fa2fda82d6152be0da25aaae194d37ff3"
SCENE_IDS = (
    "01_turn_end",
    "02_turn_not_ended",
    "03_term",
    "04_near_term",
    "05_speaker_change",
    "06_language_change",
    "07_language_pt",
    "08_three_speakers",
)


def cut_scene(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    if not 0 <= start < end <= len(audio):
        raise ValueError("Scene bounds must lie within the source assembly.")
    clip = audio[start:end][-128000:]
    return np.pad(clip, (128000 - len(clip), 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original synthetic scenes.wav assembly")
    parser.add_argument("manifest", type=Path, help="Original scene timing manifest.json")
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source does not match the documented synthetic assembly.")
    audio, rate = sf.read(args.source, dtype="int16")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("Source must be mono 16 kHz PCM audio.")
    source = json.loads(args.manifest.read_text())
    scenes = {scene["id"]: scene for scene in source["scenes"]}
    EXAMPLES.mkdir(exist_ok=True)
    manifest = {
        "source": "Synthetic speech generated with ElevenLabs",
        "model_id": source["model_id"],
        "source_sha256": SOURCE_SHA256,
        "sample_rate": 16000,
        "voices": [voice["name"] for voice in source["voices"].values()],
        "permission": "Generated on a paid plan permitting commercial use, as confirmed by owner.",
        "processing": (
            "Final eight seconds per scene; left-pad shorter scenes with digital silence."
        ),
        "scenes": [],
    }
    for name in SCENE_IDS:
        scene = scenes[name]
        start, end = scene["start_sample"], scene["end_sample"]
        path = EXAMPLES / f"{name}.wav"
        sf.write(path, cut_scene(audio, start, end), 16000, subtype="PCM_16")
        manifest["scenes"].append(
            {
                "id": name,
                "event": scene["event"],
                "start_sample": start,
                "end_sample": end,
                "left_padding_samples": max(0, 128000 - (end - start)),
                "trimmed_start_samples": max(0, end - start - 128000),
                "segments": [
                    {key: segment[key] for key in ("voice_name", "language", "text")}
                    for segment in scene["segments"]
                ],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (EXAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
