"""Audio encoder layers 1-18 as TensorRT engines, for batched window passes on an NVIDIA GPU.

Two engines that share no weights: ``part_a.engine`` maps the log-mel ``[B, 128, 800]`` of
eight-second windows to the hidden states of layers 9 and 12, and ``part_b.engine`` maps layer
12 to layer 18. An engine only loads on the GPU model and TensorRT version it was built with,
so engines are built on the host from the ONNX files (``marimba build-tensorrt``).
"""

from __future__ import annotations

import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

_INPUT_SHAPES = {"part_a": (128, 800), "part_b": (104, 896)}
ONNX_REPO = "charcot-ai/marimba-quant-tensorrt"
ONNX_REVISION = "bbccedeb2cbda2af4698f7270d657b98bf2d4dc1"


def _tensorrt() -> Any:
    try:
        import tensorrt
    except ImportError as error:
        raise ImportError("Install marimba-audio[tensorrt] to run the TensorRT encoder") from error
    return tensorrt


class _Engine:
    """One serialized engine, fed with CUDA tensors in place."""

    def __init__(self, path: Path) -> None:
        trt = _tensorrt()
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{path} does not load on this GPU with TensorRT {trt.__version__}")
        # A stream of its own: given the default stream, TensorRT adds a synchronization.
        self.stream = torch.cuda.Stream()
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode.INPUT
        self.input = next(name for name in names if self.engine.get_tensor_mode(name) == mode)
        self.outputs = [name for name in names if name != self.input]
        kinds = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.dtypes = {name: kinds[self.engine.get_tensor_dtype(name)] for name in names}
        low, _, high = self.engine.get_tensor_profile_shape(self.input, 0)
        self.max_batch = int(high[0])
        if int(low[0]) > 1:
            raise ValueError(f"{path} does not accept a batch of one")

    def __call__(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        tensor = tensor.to(self.dtypes[self.input]).contiguous()
        self.context.set_input_shape(self.input, tuple(tensor.shape))
        self.context.set_tensor_address(self.input, tensor.data_ptr())
        outputs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            buffer = torch.empty(shape, dtype=self.dtypes[name], device=tensor.device)
            self.context.set_tensor_address(name, buffer.data_ptr())
            outputs.append(buffer)
        self.stream.wait_stream(torch.cuda.current_stream())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.current_stream().wait_stream(self.stream)
        for buffer in (tensor, *outputs):
            buffer.record_stream(self.stream)
        return outputs


class TensorRTEncoder:
    """Hidden states of layers 9, 12 and 18 for a batch of log-mel windows."""

    layers = (9, 12, 18)

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self._part_a = _Engine(directory / "part_a.engine")
        self._part_b = _Engine(directory / "part_b.engine")

    def captures(self, mel: torch.Tensor, wanted: Collection[int]) -> dict[int, torch.Tensor]:
        """Layers in ``wanted``; 12 and 18 at float16, as the torch pass keeps them."""
        parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in wanted}
        for chunk in mel.split(self._part_a.max_batch):
            layer9, layer12 = self._part_a(chunk)[:2]
            states = {9: layer9, 12: layer12}
            if 18 in wanted:
                states[18] = self._part_b(layer12)[0]
            for layer in wanted:
                parts[layer].append(states[layer])
        return {
            layer: torch.cat(values).to(torch.float16 if layer in (12, 18) else torch.float32)
            for layer, values in parts.items()
        }


def build_engines(
    directory: str | Path, onnx_dir: str | Path | None = None, max_batch: int = 32
) -> dict[str, float]:
    """Builds both strongly typed float16 engines into ``directory``; returns seconds per part.

    Without ``onnx_dir``, the float16 ONNX files are read from the pinned public repository.
    """
    trt = _tensorrt()
    if onnx_dir is None:
        from huggingface_hub import snapshot_download

        onnx_dir = (
            Path(
                snapshot_download(ONNX_REPO, revision=ONNX_REVISION, allow_patterns=["onnx/fp16/*"])
            )
            / "onnx"
            / "fp16"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seconds = {}
    for part, shape in _INPUT_SHAPES.items():
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        # Weakly typed float16 changes more decisions than the torch bfloat16 pass does.
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(Path(onnx_dir) / f"{part}.onnx")):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"Could not parse {part}.onnx: {errors}")
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        sizes = (1, min(16, max_batch), max_batch)
        profile.set_shape(network.get_input(0).name, *[(batch, *shape) for batch in sizes])
        config.add_optimization_profile(profile)
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT could not build {part}")
        (directory / f"{part}.engine").write_bytes(bytes(serialized))
        seconds[part] = time.perf_counter() - started
    return seconds
