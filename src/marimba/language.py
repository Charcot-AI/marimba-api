"""ISO language codes supported by the native Qwen3-ASR decoder."""

from collections.abc import Mapping, Sequence

import numpy as np

LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def resolve_options(options: Sequence[str]) -> dict[str, str | None]:
    if not options:
        raise ValueError("Language options must not be empty")
    aliases = {name.casefold(): code for code, name in LANGUAGES.items()}
    aliases.update({code: code for code in LANGUAGES})
    resolved: dict[str, str | None] = {}
    for option in options:
        key = option.strip().casefold()
        if key == "other":
            code = None
        elif key in aliases:
            code = aliases[key]
        else:
            raise ValueError(f"Unsupported language option {option!r}")
        if code in resolved.values():
            raise ValueError("Language options must not repeat a language through aliases")
        resolved[option] = code
    return resolved


def option_probabilities(
    scores: Mapping[str, float], options: Mapping[str, str | None]
) -> dict[str, float]:
    codes = list(LANGUAGES) if None in options.values() else list(options.values())
    values = np.array([scores[code] for code in codes if code is not None], dtype=np.float64)
    weights = np.exp(values - np.max(values))
    weights /= weights.sum()
    mass = dict(zip(codes, map(float, weights), strict=True))
    listed = set(options.values()) - {None}
    return {
        option: mass[code]
        if code is not None
        else sum(value for language, value in mass.items() if language not in listed)
        for option, code in options.items()
    }
