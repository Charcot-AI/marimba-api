"""Validate units, speaker cuts and the primer bank on public speech at a 500 ms hop.

Builds a conversation from public LibriSpeech utterances of several readers, separated
by known pauses, replays it through ``run_pipeline`` and compares the result with the
known speech edges and speakers. Word times are synthetic: LibriSpeech has no word
alignment, so each utterance's words are spread over its speech by character count and
its last word ends the sentence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import islice, pairwise, permutations
from typing import Any

import numpy as np
import numpy.typing as npt

from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .protocol import SAMPLE_RATE
from .speakers import Word
from .units import UnitConfig, frame_energy_db, snap_edge

Audio = npt.NDArray[np.float32]
DATASET = "openslr/librispeech_asr"
DATASET_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
# Pauses between turns: joined, joined, joined, and one longer than the 5 s limit.
PAUSES_S = (0.0, 0.4, 0.8, 1.5, 6.0)
NOISE_LEVEL = 1e-4


@dataclass(frozen=True)
class Region:
    start_s: float
    end_s: float
    speaker: int


@dataclass(frozen=True)
class Conversation:
    audio: Audio
    regions: list[Region]
    words: list[Word]


def speech_bounds(audio: Audio, floor_db: float = 40.0) -> tuple[int, int]:
    """First and last sample of the frames within ``floor_db`` of the loudest frame."""
    energy = frame_energy_db(audio)
    loud = np.flatnonzero(energy >= energy.max() - floor_db)
    size = len(audio) // len(energy)
    return int(loud[0] * size), int((loud[-1] + 1) * size)


def build_conversation(
    clips: list[tuple[int, Audio, str]],
    pauses_s: tuple[float, ...] = PAUSES_S,
    seed: int = 0,
) -> Conversation:
    """Join ``(speaker, audio, text)`` clips with the pauses in rotation, on a noise floor."""
    rng = np.random.default_rng(seed)
    parts: list[Audio] = [
        NOISE_LEVEL * rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    ]
    position = SAMPLE_RATE
    regions: list[Region] = []
    words: list[Word] = []
    for index, (speaker, audio, text) in enumerate(clips):
        first, last = speech_bounds(audio)
        speech = audio[first:last]
        start_s = position / SAMPLE_RATE
        end_s = (position + len(speech)) / SAMPLE_RATE
        regions.append(Region(start_s, end_s, speaker))
        tokens = text.lower().split()
        weights = np.cumsum([len(token) + 1 for token in tokens], dtype=np.float64)
        marks = start_s + (end_s - start_s) * np.concatenate(
            ([0.0], weights / weights[-1])
        )
        tokens[-1] += "."
        words += [
            Word(token, float(marks[i]), float(marks[i + 1]))
            for i, token in enumerate(tokens)
        ]
        pause = round(pauses_s[index % len(pauses_s)] * SAMPLE_RATE)
        parts += [speech, NOISE_LEVEL * rng.standard_normal(pause).astype(np.float32)]
        position += len(speech) + pause
    parts.append(NOISE_LEVEL * rng.standard_normal(3 * SAMPLE_RATE).astype(np.float32))
    return Conversation(np.concatenate(parts), regions, words)


def public_clips(speakers: int, turns: int, seed: int) -> list[tuple[int, Audio, str]]:
    from datasets import Audio as AudioFeature
    from datasets import load_dataset
    from marimba.audio import decode_audio

    rows = load_dataset(
        DATASET, "clean", split="validation", streaming=True, revision=DATASET_REVISION
    ).cast_column("audio", AudioFeature(decode=False))
    by_speaker: dict[int, list[tuple[Audio, str]]] = {}
    for row in islice(rows, 400):
        audio = decode_audio(row["audio"])
        if 3 * SAMPLE_RATE <= len(audio) <= 9 * SAMPLE_RATE:
            by_speaker.setdefault(row["speaker_id"], []).append((audio, row["text"]))
    chosen = sorted(by_speaker)[:speakers]
    rng = np.random.default_rng(seed)
    clips: list[tuple[int, Audio, str]] = []
    previous = -1
    for _ in range(turns):
        speaker = int(rng.choice([s for s in range(len(chosen)) if s != previous]))
        # Every other turn has two utterances, so same-speaker joins are exercised too.
        for _ in range(1 + len(clips) % 2):
            audio, text = by_speaker[chosen[speaker]].pop(0)
            clips.append((speaker, audio, text))
        previous = speaker
    return clips


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": len(values),
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def evaluate(
    conversation: Conversation, result: PipelineResult, config: PipelineConfig
) -> dict[str, Any]:
    regions = conversation.regions
    units_config: UnitConfig = config.units
    threshold = units_config.threshold
    starts = np.array([region.start_s for region in regions])
    ends = np.array([region.end_s for region in regions])
    # Speech stretches: regions separated by less than a second count as continuous.
    onsets = [starts[0]] + [
        s for s, e in zip(starts[1:], ends[:-1], strict=True) if s - e >= 1
    ]
    offsets = [e for s, e in zip(starts[1:], ends[:-1], strict=True) if s - e >= 1] + [
        ends[-1]
    ]
    onset_lags, offset_lags = [], []
    speaking = False
    for time_s, probability in result.hops:
        if probability >= threshold and not speaking:
            earlier = [onset for onset in onsets if onset < time_s]
            if earlier and time_s - earlier[-1] < 2:
                onset_lags.append(time_s - earlier[-1])
        elif probability < threshold and speaking:
            earlier = [offset for offset in offsets if offset < time_s]
            if earlier and time_s - earlier[-1] < 2:
                offset_lags.append(time_s - earlier[-1])
        speaking = probability >= threshold

    def edge_errors(units: list[Any]) -> tuple[list[float], list[float]]:
        start_errors = [
            float(np.abs(starts - u.start / SAMPLE_RATE).min()) for u in units
        ]
        end_errors = [float(np.abs(ends - u.end / SAMPLE_RATE).min()) for u in units]
        return start_errors, end_errors

    def clipped(units: list[Any]) -> float:
        """Seconds of known speech left outside every unit."""
        lost = 0.0
        for region in regions:
            covered = sum(
                max(
                    0.0,
                    min(region.end_s, u.end / SAMPLE_RATE)
                    - max(region.start_s, u.start / SAMPLE_RATE),
                )
                for u in units
            )
            lost += region.end_s - region.start_s - covered
        return round(lost, 3)

    raw_start, raw_end = edge_errors(result.raw_units)
    snapped_start, snapped_end = edge_errors(result.units)
    # Streaming: snap each start when its opening hop arrives, without later audio.
    early_moves, buffered = [], []
    lag = units_config.samples(units_config.onset_lag_s)
    for index, (raw, unit) in enumerate(
        zip(result.raw_units, result.units, strict=True)
    ):
        heard = raw.start + lag
        energy = frame_energy_db(conversation.audio[:heard], units_config)
        lower = result.units[index - 1].end if index else 0
        early = snap_edge(energy, raw.start, lower, min(heard, raw.end), units_config)
        early_moves.append(abs(early - unit.start) / SAMPLE_RATE)
        buffered.append((heard - (unit.audio_start or 0)) / SAMPLE_RATE)
    changes = [
        left.end_s
        for left, right in pairwise(regions)
        if left.speaker != right.speaker
        and any(
            u.start / SAMPLE_RATE < left.end_s and right.start_s < u.end / SAMPLE_RATE
            for u in result.units
        )
    ]
    cuts = [cut for group in result.cuts for cut in group]
    found = sum(
        any(abs(cut.time_s - change) <= 1.0 for cut in cuts) for change in changes
    )
    correct = sum(
        any(abs(cut.time_s - change) <= 1.0 for change in changes) for cut in cuts
    )

    def truth(start_s: float, end_s: float) -> int:
        overlap = [
            max(0.0, min(end_s, r.end_s) - max(start_s, r.start_s)) for r in regions
        ]
        return regions[int(np.argmax(overlap))].speaker

    spans = [
        (s.end_s - s.start_s, truth(s.start_s, s.end_s), s.speaker)
        for s in result.spans
    ]
    labels = sorted({speaker for _, _, speaker in spans} | {r.speaker for r in regions})
    best = max(
        (
            sum(
                d
                for d, true, predicted in spans
                if mapping[labels.index(predicted)] == true
            )
            for mapping in permutations(labels)
        ),
        default=0.0,
    )
    durations = [(u.end - u.start) / SAMPLE_RATE for u in result.units]
    return {
        "hop_s": config.hop_s,
        "audio_s": round(len(conversation.audio) / SAMPLE_RATE, 2),
        "known_speech_regions": len(regions),
        "known_speakers": len({r.speaker for r in regions}),
        "onset_lag_s": {
            "assumed": units_config.onset_lag_s,
            "measured": summary(onset_lags),
        },
        "offset_lag_s": {
            "assumed": units_config.offset_lag_s,
            "measured": summary(offset_lags),
        },
        "units": {
            "count": len(result.units),
            "duration_s": summary(durations),
            "turn_ended": sum(u.turn_ended for u in result.units),
            "start_error_s": {
                "raw": summary(raw_start),
                "snapped": summary(snapped_start),
            },
            "end_error_s": {"raw": summary(raw_end), "snapped": summary(snapped_end)},
            "speech_outside_units_s": {
                "raw": clipped(result.raw_units),
                "snapped": clipped(result.units),
                "padded": clipped(
                    [
                        type(u)(u.audio_start or u.start, u.audio_end or u.end)
                        for u in result.units
                    ]
                ),
            },
        },
        "streaming": {
            "start_snapped_at_opening_hop_vs_final_s": summary(early_moves),
            "buffered_audio_needed_at_opening_hop_s": summary(buffered),
        },
        "speaker_cuts": {
            "known_changes_inside_units": len(changes),
            "found_within_1s": found,
            "cuts": len(cuts),
            "cuts_at_a_known_change": correct,
            "cuts_at_sentence_end": sum(cut.at_sentence_end for cut in cuts),
        },
        "primer_bank": {
            "speakers_created": len({speaker for _, _, speaker in spans}),
            "spans": len(spans),
            "duration_weighted_accuracy": round(
                best / max(sum(d for d, _, _ in spans), 1e-9), 3
            ),
        },
    }


def transcriber(conversation: Conversation) -> Any:
    def transcribe(audio: Audio, start_s: float) -> list[Word]:
        end_s = start_s + len(audio) / SAMPLE_RATE
        return [
            w
            for w in conversation.words
            if start_s <= (w.start_s + w.end_s) / 2 < end_s
        ]

    return transcribe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--turns", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from marimba import Marimba, Question

    conversation = build_conversation(
        public_clips(args.speakers, args.turns, args.seed)
    )
    model = Marimba.from_pretrained(device=args.device)
    config = PipelineConfig(hop_s=args.hop)
    result = run_pipeline(
        model,
        Question,
        conversation.audio,
        transcribe=transcriber(conversation),
        config=config,
    )
    report = {"dataset": DATASET, "runtime": model.runtime.to_dict()}
    report.update(evaluate(conversation, result, config))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
