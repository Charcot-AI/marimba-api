"""Capability-driven dtype selection without requiring a GPU or torch install."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marimba import runtime
from marimba.cli import main


@pytest.fixture
def mock_torch(monkeypatch):
    def install(cuda, bf16, capability):
        cuda_api = SimpleNamespace(
            is_available=Mock(return_value=cuda),
            is_bf16_supported=Mock(return_value=bf16),
            get_device_capability=Mock(return_value=capability),
            mem_get_info=Mock(return_value=(24 * 1024**3, 32 * 1024**3)),
        )
        torch = SimpleNamespace(
            cuda=cuda_api,
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        monkeypatch.setattr(
            runtime,
            "importlib",
            SimpleNamespace(
                util=SimpleNamespace(find_spec=lambda name: object() if name == "torch" else None),
                import_module=lambda name: torch,
            ),
        )
        return cuda_api

    return install


@pytest.mark.parametrize(
    ("cuda", "bf16", "capability", "device", "expected"),
    [
        (True, True, (8, 9), "auto", "bfloat16"),
        (True, False, (7, 5), "auto", "float16"),
        (True, False, (5, 2), "auto", "float32"),
        (False, False, (0, 0), "auto", "float32"),
        (True, True, (8, 9), "cpu", "float32"),
    ],
)
def test_auto_dtype(mock_torch, cuda, bf16, capability, device, expected):
    cuda_api = mock_torch(cuda, bf16, capability)
    selected = runtime.select_backend(backend="torch", device=device)
    assert selected.dtype == expected
    assert selected.to_dict()["dtype"] == expected
    if not cuda:
        cuda_api.is_bf16_supported.assert_not_called()
        cuda_api.get_device_capability.assert_not_called()


def test_explicit_dtype_overrides_auto(mock_torch):
    mock_torch(True, True, (8, 9))
    assert runtime.select_backend(dtype="float32").dtype == "float32"
    assert runtime.select_backend(dtype="float16").dtype == "float16"


@pytest.mark.parametrize(
    ("bf16", "capability", "expected"),
    [(True, (8, 9), "bfloat16"), (False, (7, 5), "float16"), (False, (5, 2), "float32")],
)
def test_doctor_reports_cuda_capabilities_and_resolved_dtype(
    mock_torch, capsys, bf16, capability, expected
):
    mock_torch(True, bf16, capability)
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cuda_bf16_supported"] is bf16
    assert report["cuda_fp16_supported"] is (capability >= (5, 3))
    assert report["selection"]["device"] == "cuda"
    assert report["selection"]["dtype"] == expected


@pytest.mark.parametrize("memory", [None, 4 * 1024**3, 16 * 1024**3])
def test_mps_auto_policy_unchanged(monkeypatch, memory):
    monkeypatch.setattr(
        runtime,
        "detect_runtime",
        lambda: {
            "extras": {"torch": True},
            "cuda": False,
            "mps": True,
            "providers": (),
            "free_memory": memory,
        },
    )
    selected = runtime.select_backend()
    assert selected.device == "mps"
    assert selected.dtype == ("float16" if memory is None or memory < 8 * 1024**3 else "float32")
