"""Exact normalized token-sequence scoring across shared decoder hypotheses."""

import re
import unicodedata
from collections.abc import Sequence


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", unaccented).split())


def exact_fraction(term: str, hypotheses: Sequence[str]) -> float:
    target = normalize(term)
    if not target or not hypotheses:
        raise ValueError("A nonempty normalized term and decoder hypotheses are required")
    return sum(f" {target} " in f" {normalize(text)} " for text in hypotheses) / len(hypotheses)
