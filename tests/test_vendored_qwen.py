from importlib.metadata import requires
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement


def test_torch_extra_direct_dependencies():
    selected = [Requirement(value) for value in requires("marimba-audio")]
    names = {r.name for r in selected if r.marker is None or r.marker.evaluate({"extra": "torch"})}
    assert "transformers" in names and "accelerate" in names
    assert not names & {"qwen-asr", "gradio", "fastapi", "starlette"}


def test_prompt_and_output_parser():
    pytest.importorskip("torch")
    from marimba._qwen import FrozenASR
    from marimba._vendor.qwen_asr.utils import parse_asr_output

    def template(messages, **kwargs):
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return "prompt"

    stub = SimpleNamespace(processor=SimpleNamespace(apply_chat_template=template))
    assert FrozenASR._build_text_prompt(stub, "", None) == "prompt"
    assert FrozenASR._build_text_prompt(stub, "", "English") == "promptlanguage English<asr_text>"
    assert parse_asr_output("language English<asr_text>hello") == ("English", "hello")
    assert parse_asr_output("language None<asr_text>") == ("", "")
    assert parse_asr_output(" hello ") == ("", "hello")
    assert parse_asr_output("a" * 25) == ("", "a")
