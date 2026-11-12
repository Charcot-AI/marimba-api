"""Present scheduler payloads without changing their decisions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

ROLLING = {
    "voice_activity": "Speech",
    "speaker_change": "Speaker changed",
    "speaker_count": "People",
}
UTTERANCE = {"turn_end", "language", "term"}


def card_values(cards: list[Any], latest: dict, utterances: Iterable[dict]) -> list[str]:
    history = list(utterances)
    result = []
    for card in cards:
        decision = latest.get(card.task)
        text = f"### {card.title}\n\n"
        if not decision:
            text += (
                "Waiting for a closed speech run."
                if card.task in UTTERANCE
                else "Waiting for audio."
            )
        elif decision["value"] is None:
            text += f"No prediction · {decision['status'].replace('_', ' ')}"
        else:
            probability = decision.get("probability")
            text += f"**{decision['value']}**"
            if probability is not None:
                label = "raw N-best fraction" if card.task == "term" else "probability"
                text += f" · {label} {probability:.4f}"
            text += f"\n\n{decision['window_start_s']:.2f}–{decision['window_end_s']:.2f} s"
        if card.task in UTTERANCE:
            for item in history:
                if item["task"] == card.task:
                    probability = item.get("probability")
                    score = f" ({probability:.4f})" if probability is not None else ""
                    text += (
                        f"\n\n- {item['window_start_s']:.2f}–{item['window_end_s']:.2f} s: "
                        f"{item['value']}{score}"
                    )
        result.append(text)
    return result


def present(events: list[dict], cards: list[Any]) -> tuple:
    latest, history, rows = {}, [], []
    for event in events:
        for decision in event.get("decisions", []):
            task = decision["task"]
            latest[task] = decision
            if task in UTTERANCE:
                history.append(decision)
            probability = decision.get("probability")
            if task in ROLLING and probability is not None:
                rows.append(
                    {
                        "window_end_s": decision["window_end_s"],
                        "probability": probability,
                        "card": ROLLING[task],
                    }
                )
    plot = pd.DataFrame(rows, columns=["window_end_s", "probability", "card"])
    return (*card_values(cards, latest, history), plot)
