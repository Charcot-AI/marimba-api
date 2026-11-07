"""Optional runtime discovery and explicit backend selection."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    backend: str
    device: str
    dtype: str
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "providers": list(self.providers)}


def detect_runtime() -> dict[str, Any]:
    extras = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "onnxruntime", "datasets", "gradio", "transformers", "torchcodec")
    }
    info: dict[str, Any] = {
        "os": platform.system(),
        "cpu": platform.machine(),
        "extras": extras,
        "cuda": False,
        "mps": False,
        "providers": (),
        "free_memory": None,
    }
    try:
        info["free_memory"] = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    if extras["torch"]:
        torch = importlib.import_module("torch")
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = torch.backends.mps.is_available()
        if info["cuda"]:
            info["cuda_free_memory"] = torch.cuda.mem_get_info()[0]
    if extras["onnxruntime"]:
        info["providers"] = tuple(importlib.import_module("onnxruntime").get_available_providers())
    return info


def select_backend(backend: str = "auto", device: str = "auto", dtype: str = "auto") -> RuntimeInfo:
    if backend not in ("auto", "torch", "onnx"):
        raise ValueError("backend must be auto, torch, or onnx")
    if device not in ("auto", "cpu", "cuda", "mps", "coreml"):
        raise ValueError("Unsupported device")
    if dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype")
    info = detect_runtime()
    candidates: list[tuple[str, str, tuple[str, ...]]] = []
    if info["extras"]["torch"]:
        if info["cuda"]:
            candidates.append(("torch", "cuda", ()))
        if info["mps"]:
            candidates.append(("torch", "mps", ()))
    for provider, target in (
        ("CoreMLExecutionProvider", "coreml"),
        ("CUDAExecutionProvider", "cuda"),
        ("CPUExecutionProvider", "cpu"),
    ):
        if provider in info["providers"]:
            candidates.append(("onnx", target, (provider,)))
    if info["extras"]["torch"]:
        candidates.append(("torch", "cpu", ()))
    for engine, target, providers in candidates:
        if backend not in ("auto", engine) or device not in ("auto", target):
            continue
        memory = info.get("cuda_free_memory") if target == "cuda" else info["free_memory"]
        precision = dtype
        if dtype == "auto":
            precision = (
                "float16"
                if target in ("cuda", "mps") and (memory is None or memory < 8 * 1024**3)
                else "float32"
            )
        result = RuntimeInfo(engine, target, precision, providers)
        logging.getLogger(__name__).info("Selected runtime: %s", result)
        return result
    raise RuntimeError("No matching inference runtime. Install marimba[torch] or marimba[onnx].")


def synchronize(runtime: RuntimeInfo) -> None:
    if runtime.backend == "torch" and runtime.device in ("cuda", "mps"):
        torch = importlib.import_module("torch")
        getattr(torch, runtime.device).synchronize()
