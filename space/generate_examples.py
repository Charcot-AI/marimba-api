"""Generate deterministic synthetic examples with eSpeak NG and FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

EXAMPLES = Path(__file__).parent / "examples"
UTTERANCES = {
    "english_marimba": [("en-us", "The marimba makes a warm sound. I enjoy playing music.", 3.9)],
    "portuguese_near_word": [
        ("pt-br", "A marina fica perto da praia. Hoje o tempo está bonito.", 3.7)
    ],
    "spanish_two_speakers": [
        ("es", "Vamos a escuchar música esta tarde.", 0.1),
        ("es+f3", "Sí, me gusta mucho la marimba.", 5.6),
    ],
}


def main() -> None:
    EXAMPLES.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        for name, utterances in UTTERANCES.items():
            inputs = []
            filters = []
            for index, (voice, text, offset) in enumerate(utterances):
                path = Path(directory) / f"{name}-{index}.wav"
                subprocess.run(
                    ["espeak-ng", "-v", voice, "-s", "175", "-w", str(path), text], check=True
                )
                inputs.extend(["-i", str(path)])
                filters.append(f"[{index}:a]adelay={int(offset * 1000)}:all=1[a{index}]")
            streams = "".join(f"[a{index}]" for index in range(len(utterances)))
            filters.append(
                f"{streams}amix=inputs={len(utterances)}:normalize=0,apad,atrim=duration=8[out]"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    ";".join(filters),
                    "-map",
                    "[out]",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(EXAMPLES / f"{name}.wav"),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
