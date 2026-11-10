"""Typed questions and supported English routing templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["noul", "score", "choice"]

_TEMPLATES = {
    "same_speaker": r"(is (this|it) the same speaker|are (they|these) the same speaker)",
    "speaker_count": r"how many speakers( are there| are speaking)?",
    "speaker_change": (
        r"(did the speaker change|has the speaker changed|is this a different speaker)"
    ),
    "voice_activity": r"(is (someone|anyone) (speaking|talking)|is there speech)",
    "turn_end": (
        r"(has (the speaker|someone) finished( speaking)?|"
        r"is (the )?turn (over|finished)|is it (the )?end of (the )?turn)"
    ),
    "term": r"""did (someone|anyone) say ["“'].+?["”']""",
    "language": r"(what|which) language( is (being )?spoken| is this)?",
    "pii": (
        r"(does (this|the) audio contain (any )?(personal|personally identifiable) "
        r"(information|data)|is there (any )?(pii|personal (information|data)))"
    ),
}


def matches_template(task: str, text: str) -> bool:
    pattern = _TEMPLATES.get(task)
    return (
        pattern is not None
        and re.fullmatch(pattern, text.lower().strip().rstrip("?.!")) is not None
    )


class UnsupportedQuestion(ValueError):
    """The question has no supported routing template."""


@dataclass(frozen=True)
class Question:
    text: str
    kind: Kind
    options: tuple[str, ...] | None = None
    task: str | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.text.strip() or not 0 <= self.threshold <= 1:
            raise ValueError("Question text and a threshold between zero and one are required")
        if self.kind not in ("noul", "score", "choice"):
            raise ValueError("Invalid question kind")
        if self.task is None:
            for task in _TEMPLATES:
                if matches_template(task, self.text):
                    object.__setattr__(self, "task", task)
                    break
            else:
                raise UnsupportedQuestion(
                    f"No routing template for {self.text!r}; pass task explicitly"
                )
        if self.task in ("same_speaker", "speaker_change") and self.kind not in ("noul", "score"):
            raise UnsupportedQuestion(f"Task {self.task} supports noul or score")
        if self.task == "speaker_count" and (
            self.kind != "choice"
            or len(self.options or ()) != 4
            or set(self.options or ()) != {"0", "1", "2", "3+"}
        ):
            raise UnsupportedQuestion("Speaker count requires choice options 0, 1, 2, 3+")
        if self.kind == "choice":
            if not self.options or len(set(self.options)) != len(self.options):
                raise ValueError("Choice questions require distinct, nonempty options")
            object.__setattr__(self, "options", tuple(self.options))

    @property
    def term(self) -> str | None:
        match = re.search(r"""(?:"([^"]+)"|“([^”]+)”|'([^']+)')""", self.text)
        return (
            next(group for group in match.groups() if group is not None)
            if (self.task == "term" and match)
            else None
        )

    @classmethod
    def noul(cls, text: str, *, task: str | None = None, threshold: float = 0.5) -> Question:
        return cls(text, "noul", task=task, threshold=threshold)

    @classmethod
    def score(cls, text: str, *, task: str | None = None, threshold: float = 0.5) -> Question:
        return cls(text, "score", task=task, threshold=threshold)

    @classmethod
    def choice(
        cls, text: str, options: tuple[str, ...] | list[str], *, task: str | None = None
    ) -> Question:
        return cls(text, "choice", tuple(options), task)
