"""Opt-in, report-only encoder/decoder measurements on a caller-owned corpus."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import time
import traceback
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from qwen_asr.inference.utils import parse_asr_output

from marimba import Marimba
from marimba.lexical import exact_fraction
from marimba.readout import pool_layer, sigmoid


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def auroc(labels: list[int], scores: list[float]) -> float:
    labels_array = np.asarray(labels)
    values = np.asarray(scores)
    positives, negatives = values[labels_array == 1], values[labels_array == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("AUROC requires both classes")
    pairs = positives[:, None] - negatives[None, :]
    return float(np.mean((pairs > 0) + 0.5 * (pairs == 0)))


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(function: Any, device: str) -> tuple[Any, float]:
    sync(device)
    start = time.perf_counter()
    result = function()
    sync(device)
    return result, (time.perf_counter() - start) * 1000


def raw_turn(engine: Any, features: Any, mask: Any) -> float:
    pooled = pool_layer(features[18], mask)
    weights = engine.readout
    return float(
        ((pooled - weights["mean"]) / weights["scale"]) @ weights["coef"] + weights["intercept"][0]
    )


def generate(
    engine: Any, audio: Any, prefix: Any, beams: tuple[int, ...], tokens: int
) -> list[str]:
    prompt = engine.asr._build_text_prompt(context="", force_language=None)
    result = []
    with torch.inference_mode():
        for count in beams:
            inputs = engine._inputs(prompt, audio)
            engine.thinker.rope_deltas = prefix["rope_deltas"].clone()
            cache = deepcopy(prefix["cache"])
            if count > 1:
                cache.batch_repeat_interleave(count)
            output = engine.asr.model.generate(
                **inputs,
                past_key_values=cache,
                num_beams=count,
                num_return_sequences=count,
                max_new_tokens=tokens,
                do_sample=False,
                length_penalty=1.0,
            )
            decoded = engine.processor.batch_decode(
                output.sequences[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            result.extend(parse_asr_output(text)[1] for text in decoded)
    return result


class AudioExport(torch.nn.Module):
    """Fixed eight-second, batch-one graph; native L9/L18 and projected prefix."""

    def __init__(self, tower: Any):
        super().__init__()
        self.tower = tower
        self.taps: dict[int, Any] = {}
        self.handles = [
            tower.layers[layer - 1].register_forward_hook(self.capture(layer)) for layer in (9, 18)
        ]

    def capture(self, layer: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            self.taps[layer] = output[0]

        return hook

    def forward(self, mel: Any) -> Any:
        lengths = torch.tensor([800], device=mel.device)
        output = self.tower(mel, feature_lens=lengths).last_hidden_state
        return output, self.taps[9], self.taps[18]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def export(engine: Any, path: Path) -> None:
    wrapper = AudioExport(engine.thinker.audio_tower)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (torch.zeros((128, 800)),),
                str(path),
                input_names=["mel"],
                output_names=["audio", "l9", "l18"],
                opset_version=17,
                dynamo=False,
            )
        importlib.import_module("onnx").checker.check_model(str(path))
    finally:
        wrapper.close()


def ort_encoder(engine: Any, path: Path, threads: int) -> Any:
    ort = importlib.import_module("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    def encode(window: Any) -> dict[str, Any]:
        inputs = engine.processor.feature_extractor(
            [window],
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="np",
            truncation=False,
        )
        audio, l9, l18 = session.run(None, {"mel": inputs["input_features"][0]})
        return {
            "features": {9: l9, 18: l18.astype(np.float16).astype(np.float32)},
            "audio_features": audio,
        }

    return encode


def measure(args: argparse.Namespace, variant: str, threads: int) -> dict[str, Any]:
    corpus = json.loads(Path(args.corpus).with_suffix(".json").read_text())
    arrays = np.load(args.corpus, allow_pickle=False)
    counts = {
        lane: sum(row["lane"] == lane for row in corpus["records"])
        for lane in ("turn_end", "decoder")
    }
    if min(counts.values()) < 100:
        raise ValueError("At least 100 turn and 100 decoder parity clips are required")
    if any(row.get("split", "train") not in ("train", "validation") for row in corpus["records"]):
        raise ValueError("Only train/validation or synthetic clips are permitted")
    torch.set_num_threads(threads)
    dtype = (
        "float16" if variant == "cuda_fp16" else "bfloat16" if variant == "cuda_bf16" else "float32"
    )
    model = Marimba.from_pretrained(
        backend="torch", device=args.device, dtype=dtype, revision=args.revision
    )
    engine = model._engine
    config = {
        "variant": variant,
        "threads": threads,
        "device": args.device,
        "dtype": dtype,
        "head_dtype": "float32",
        "attention": "sdpa",
        "revision": model.revision,
        "beams": [1, 4],
        "nbest": 5,
        "max_new_tokens": 256,
    }
    if variant == "encoder_dynamic_int8":
        engine.thinker.audio_tower = torch.ao.quantization.quantize_dynamic(
            engine.thinker.audio_tower, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
    if variant.startswith("ort_"):
        path = Path(args.output).parent / "audio_tower_fp32.onnx"
        if not path.exists():
            export(engine, path)
        if variant == "ort_int8":
            quant = importlib.import_module("onnxruntime.quantization")
            target = path.with_name("audio_tower_int8.onnx")
            if not target.exists():
                quant.quantize_dynamic(
                    str(path),
                    str(target),
                    weight_type=quant.QuantType.QInt8,
                    op_types_to_quantize=["MatMul", "Gemm"],
                )
            path = target
        engine.encode = ort_encoder(engine, path, threads)
    if variant.endswith("beam2"):
        config.update(beams=[1, 2], nbest=3, max_new_tokens=128)
    elif variant.endswith("greedy"):
        config.update(beams=[1], nbest=1, max_new_tokens=64)
    timing: dict[str, list[float]] = {
        name: [] for name in ["encoder", "readout_heads", "prefix", "lid", "term", "decoder_total"]
    }
    output = {
        "config": config,
        "status": "running",
        "timing_ms": {},
        "records": [],
        "warmup": args.warmup,
        "corpus_revisions": corpus["revisions"],
    }
    options = ["en", "pt", "es", "fr", "de"]
    first_decoder = next(i for i, row in enumerate(corpus["records"]) if row["lane"] == "decoder")
    for _ in range(args.warmup):
        data = engine.encode(arrays["audio"][first_decoder])
        engine.turn_end(data["features"], arrays["mask"][first_decoder])
        engine.head_score(data["features"], arrays["mask"][first_decoder], "Is someone speaking?")
        prefix = engine.prefill(data["audio_features"])
        engine.language_scores(data["audio_features"], options, prefix)
        generate(
            engine, data["audio_features"], prefix, tuple(config["beams"]), config["max_new_tokens"]
        )
    for index, row in enumerate(corpus["records"]):
        window, mask = arrays["audio"][index], arrays["mask"][index]
        data, ms = timed(partial(engine.encode, window), args.device)
        timing["encoder"].append(ms)

        def heads(data: Any = data, mask: Any = mask) -> tuple[Any, float, float]:
            return (
                engine.turn_end(data["features"], mask),
                engine.head_score(data["features"], mask, "Is someone speaking?"),
                raw_turn(engine, data["features"], mask),
            )

        (turn, vad_logit, turn_logit), ms = timed(heads, args.device)
        timing["readout_heads"].append(ms)
        record = {
            "index": index,
            "turn_prob": turn[1],
            "turn_logit": turn_logit,
            "vad_prob": sigmoid(vad_logit),
            "vad_logit": vad_logit,
        }
        if row["lane"] == "decoder":
            prefix, pms = timed(partial(engine.prefill, data["audio_features"]), args.device)
            scores, lms = timed(
                partial(engine.language_scores, data["audio_features"], options, prefix),
                args.device,
            )
            texts, tms = timed(
                partial(
                    generate,
                    engine,
                    data["audio_features"],
                    prefix,
                    tuple(config["beams"]),
                    config["max_new_tokens"],
                ),
                args.device,
            )
            timing["prefix"].append(pms)
            timing["lid"].append(lms)
            timing["term"].append(tms)
            timing["decoder_total"].append(pms + lms + tms)
            record.update(
                language=max(scores, key=scores.get),
                language_log_scores=scores,
                term_prob=[exact_fraction(row[key], texts) for key in ["positive", "negative"]],
                transcripts=texts,
            )
        output["records"].append(record)
        if index % 10 == 0:
            print(variant, index, flush=True)
            Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    output.update(
        status="completed", timing_ms={key: summary(value) for key, value in timing.items()}
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="NPZ audio/mask plus adjacent JSON records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "cpu_fp32",
            "cpu_tuned",
            "encoder_dynamic_int8",
            "ort_fp32",
            "ort_int8",
            "cpu_beam2",
            "cpu_greedy",
            "cuda_fp32",
            "cuda_fp16",
            "cuda_bf16",
            "cuda_beam2",
            "cuda_greedy",
        ],
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
    }
    try:
        result = measure(args, args.variant, args.threads)
    except Exception as error:
        result = {
            "status": "failed",
            "config": vars(args),
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(result["traceback"], flush=True)
    result["environment"] = metadata
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("MATRIX_DONE", args.variant, result["status"], flush=True)


if __name__ == "__main__":
    main()
