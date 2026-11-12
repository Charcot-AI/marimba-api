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
    cadence: str = "every 250 ms"
    seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CARDS = (
    Card("turn_end", "Turn ended", "Has the speaker finished?", "noul"),
    Card(
        "voice_activity",
        "Speech (last 2 s)",
        "Is someone speaking?",
        "noul",
        context_seconds=2.0,
        seconds=2.0,
    ),
    Card("speaker_change", "Speaker changed", "Has the speaker changed?", "score"),
    Card(
        "speaker_count",
        "People (last 8 s)",
        "How many people?",
        "choice",
        ("0", "1", "2", "3+"),
    ),
    Card(
        "language",
        "Language",
        "Which language is spoken?",
        "choice",
        ("en", "pt", "es"),
    ),
    Card("term", "Marimba spoken", 'Was "marimba" spoken?', "noul"),
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
    questions = []
    for card in cards:
        kwargs: dict[str, Any] = {"task": card.task}
        if card.seconds is not None:
            kwargs["seconds"] = card.seconds
        constructor = getattr(question_class, card.kind)
        questions.append(
            constructor(card.text, card.options, **kwargs)
            if card.kind == "choice"
            else constructor(card.text, **kwargs)
        )
    return questions
