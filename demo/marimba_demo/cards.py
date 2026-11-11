"""The questions shown when the loaded model advertises support."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Card:
    task: str
    title: str
    text: str
    kind: str
    options: tuple[str, ...] = ()
    status: str = "unavailable"
    evaluation: str = ""
    uncalibrated: bool = False
    lane: str = "encoder"
    context_seconds: float = 8.0
    cadence_seconds: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CARDS = (
    Card(
        "turn_end", "Turn completion", "Has the speaker finished their turn?", "score"
    ),
    Card("voice_activity", "Speech present", "Is someone speaking?", "noul"),
    Card(
        "language",
        "Language",
        "What language is being spoken?",
        "choice",
        ("en", "pt", "es"),
    ),
    Card("term", "Heard marimba", 'Did someone say "marimba"?', "noul"),
    Card("same_speaker", "Same speaker", "Is this the same speaker?", "noul"),
    Card("speaker_change", "Speaker change", "Did the speaker change?", "score"),
    Card(
        "speaker_count",
        "Speakers",
        "How many speakers?",
        "choice",
        ("0", "1", "2", "3+"),
    ),
)


def available_cards(model: Any) -> list[Card]:
    cards = []
    for card in CARDS:
        capability = model.capabilities.get(card.task)
        if capability is None or capability.status == "unavailable":
            continue
        if card.kind not in capability.kinds:
            continue
        evaluation = getattr(capability, "evaluation", "") or ""
        cards.append(
            Card(
                **{
                    **asdict(card),
                    "status": capability.status,
                    "evaluation": evaluation,
                    "uncalibrated": "uncalibrat" in evaluation.casefold(),
                }
            )
        )
    return cards


def questions_for(cards: list[Card], question_class: Any) -> list[Any]:
    return [
        getattr(question_class, card.kind)(card.text, card.options, task=card.task)
        if card.kind == "choice"
        else getattr(question_class, card.kind)(card.text, task=card.task)
        for card in cards
    ]
