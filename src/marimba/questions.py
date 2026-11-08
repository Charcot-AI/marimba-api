"""Typed questions and supported English routing templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["noul", "score", "choice"]


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
        if self.kind == "choice":
            if not self.options or len(set(self.options)) != len(self.options):
                raise ValueError("Choice questions require distinct, nonempty options")
            object.__setattr__(self, "options", tuple(self.options))
        if self.task is None:
            text = self.text.lower().strip().rstrip("?.!")
            templates = {
                "voice_activity": r"(is (someone|anyone) (speaking|talking)|is there speech)",
                "turn_end": (
                    r"(has (the speaker|someone) finished( speaking)?|"
                    r"is (the )?turn (over|finished)|is it (the )?end of (the )?turn)"
                ),
                "term": r"""did (someone|anyone) say ["“'].+?["”']""",
                "language": r"(what|which) language( is (being )?spoken| is this)?",
                "same_speaker": (
                    r"(is (this|it) the same speaker|are (they|these) the same speaker)"
                ),
                "speaker_count": r"how many speakers( are there| are speaking)?",
            }
            for task, pattern in templates.items():
                if re.fullmatch(pattern, text):
                    object.__setattr__(self, "task", task)
                    break
            else:
                raise UnsupportedQuestion(
                    f"No routing template for {self.text!r}; pass task explicitly"
                )

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
