import numpy as np
import pytest

from marimba.language import LANGUAGES, option_probabilities, resolve_options


def test_names_iso_and_other_mass():
    options = resolve_options(["English", "Portuguese", "Spanish", "Other"])
    result = option_probabilities(dict.fromkeys(LANGUAGES, 0.0), options)
    assert result["English"] == pytest.approx(1 / 30)
    assert result["Other"] == pytest.approx(27 / 30)
    assert sum(result.values()) == pytest.approx(1)
    assert resolve_options(["EN", " Portuguese "]) == {"EN": "en", " Portuguese ": "pt"}
    iso = option_probabilities({"en": 2.0, "pt": 1.0}, resolve_options(["en", "pt"]))
    names = option_probabilities({"en": 2.0, "pt": 1.0}, resolve_options(["English", "Portuguese"]))
    np.testing.assert_allclose(list(iso.values()), list(names.values()))


@pytest.mark.parametrize("options", [[], ["English", "en"], ["Other", "other"], ["Klingon"]])
def test_invalid_language_options(options):
    with pytest.raises(ValueError):
        resolve_options(options)
