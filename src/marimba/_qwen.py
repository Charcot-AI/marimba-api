"""Optional torch engine with one shared audio-tower pass per window."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextvars import ContextVar
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

from ._head import SpeechReadout
from ._vendor.qwen_asr.configuration_qwen3_asr import Qwen3ASRConfig
from ._vendor.qwen_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from ._vendor.qwen_asr.processing_qwen3_asr import Qwen3ASRProcessor
from ._vendor.qwen_asr.utils import parse_asr_output
from .audio import AudioArray, Mask
from .language import LANGUAGES
from .lexical import exact_fraction
from .readout import pool_layer, predict_readout
from .runtime import RuntimeInfo
from .weights import ENCODER_REPO, ENCODER_REVISION


class FrozenASR:
    def __init__(self, path: str, runtime: RuntimeInfo) -> None:
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            path,
            config=Qwen3ASRConfig.from_pretrained(path),
            dtype=getattr(torch, runtime.dtype),
            device_map=runtime.device,
            attn_implementation="sdpa",
        )
        self.processor: Any = Qwen3ASRProcessor.from_pretrained(path, fix_mistral_regex=True)

    def _build_text_prompt(self, context: str, force_language: str | None) -> str:
        messages = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt: str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return prompt + f"language {force_language}<asr_text>" if force_language else prompt


class QwenEngine:
    def __init__(self, runtime: RuntimeInfo, paths: dict[str, Path]) -> None:
        self.asr = FrozenASR(snapshot_download(ENCODER_REPO, revision=ENCODER_REVISION), runtime)
        self.asr.model.eval().requires_grad_(False)
        self.processor = self.asr.processor
        self.processor.tokenizer.padding_side = "left"
        self.thinker = self.asr.model.thinker
        self._captures: ContextVar[Any] = ContextVar("audio_captures", default=None)
        self.head = (
            SpeechReadout(paths, runtime.device)
            if "speech_head/model.safetensors" in paths
            else None
        )
        layers = (9, 18) if self.head is not None else (18,)
        self._hooks = [
            self.thinker.audio_tower.layers[layer - 1].register_forward_hook(
                partial(self._capture, layer)
            )
            for layer in layers
        ]
        self.readout = load_file(paths["turn_end/readout.safetensors"])
        config = json.loads(paths["turn_end/config.json"].read_text())
        if config != {
            "layer": 18,
            "pooling": "tail",
            "tail_seconds": 2.0,
            "window_seconds": 8.0,
            "sampling_rate": 16000,
            "calibration": "platt",
        }:
            raise ValueError("Unsupported turn-end readout configuration")

    def _capture(self, layer: int, module: Any, inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captures = self._captures.get()
        if captures is not None:
            hidden = hidden.detach()
            if layer == 18:
                hidden = hidden.to(torch.float16)
            captures[layer] = hidden.float().cpu().numpy().copy()

    def encode(self, window: AudioArray) -> Any:
        features = self.processor.feature_extractor(
            [window],
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=False,
        )
        inputs = features["input_features"].to(self.asr.model.device, self.asr.model.dtype)
        attention_mask = features["attention_mask"].to(self.asr.model.device)
        captures: dict[int, Any] = {}
        token = self._captures.set(captures)
        try:
            with torch.inference_mode():
                encoded = self.thinker.get_audio_features(
                    inputs, feature_attention_mask=attention_mask
                )
        finally:
            self._captures.reset(token)
        return {"features": captures, "audio_features": encoded.detach().float().cpu().numpy()}

    def _inputs(self, text: str, encoded: Any) -> dict[str, Any]:
        encoded = torch.from_numpy(np.array(encoded, copy=True))
        expanded = self.processor.replace_multimodal_special_tokens([text], iter([len(encoded)]))
        batch = self.processor.tokenizer(expanded, return_tensors="pt", padding=True)
        batch = batch.to(self.asr.model.device)
        embeddings = self.thinker.get_input_embeddings()(batch["input_ids"])
        audio_mask = self.thinker.get_placeholder_mask(batch["input_ids"], inputs_embeds=embeddings)
        embeddings = embeddings.masked_scatter(
            audio_mask, encoded.to(device=embeddings.device, dtype=embeddings.dtype)
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "inputs_embeds": embeddings,
        }

    def turn_end(self, features: Any, mask: Mask) -> tuple[float, float]:
        return predict_readout(self.readout, pool_layer(features[18], mask))

    def head_score(self, features: Any, mask: Mask, text: str) -> float:
        if self.head is None:
            raise ValueError("No speech head weights are installed")
        return self.head.score(torch.from_numpy(np.array(features[9], copy=True)), mask, text)

    def prefill(self, encoded: Any) -> dict[str, Any]:
        prompt = self.asr._build_text_prompt(context="", force_language=None)
        inputs = self._inputs(prompt, encoded)
        length = inputs["input_ids"].shape[1] - 1
        with torch.inference_mode():
            output = self.thinker(
                **{key: value[:, :length] for key, value in inputs.items()}, use_cache=True
            )
        return {
            "cache": output.past_key_values,
            "length": length,
            "ids": inputs["input_ids"][:, :length],
            "rope_deltas": self.thinker.rope_deltas.clone(),
        }

    def _cached_inputs(self, inputs: dict[str, Any], prefix: Any) -> dict[str, Any]:
        length = prefix["length"]
        if not torch.equal(inputs["input_ids"][:, :length], prefix["ids"]):
            raise ValueError("Decoder text does not share the encoded audio prefix")
        self.thinker.rope_deltas = prefix["rope_deltas"].clone()
        return {
            **inputs,
            "input_ids": inputs["input_ids"][:, length:],
            "inputs_embeds": inputs["inputs_embeds"][:, length:],
            "past_key_values": deepcopy(prefix["cache"]),
            "cache_position": torch.arange(
                length, inputs["input_ids"].shape[1], device=inputs["input_ids"].device
            ),
        }

    def language_scores(
        self, encoded: Any, options: Sequence[str], prefix: Any = None
    ) -> dict[str, float]:
        base = self.asr._build_text_prompt(context="", force_language=None) + "language"
        tokenizer = self.processor.tokenizer
        prefix_ids = tokenizer.encode(base, add_special_tokens=False)
        scores = {}
        with torch.inference_mode():
            for option in options:
                if option not in LANGUAGES:
                    raise ValueError(
                        f"Unsupported language code {option!r}; supported: {', '.join(LANGUAGES)}"
                    )
                text = base + " " + LANGUAGES[option]
                full = tokenizer.encode(text, add_special_tokens=False)
                if full[: len(prefix_ids)] != prefix_ids:
                    raise ValueError("Native language prompt has an unstable token prefix")
                suffix = full[len(prefix_ids) :]
                inputs = self._inputs(text, encoded)
                end = inputs["input_ids"].shape[1]
                start = end - len(suffix)
                if inputs["input_ids"][0, start:end].tolist() != suffix:
                    raise ValueError("Native language suffix token positions do not match")
                offset = prefix["length"] if prefix is not None else 0
                if prefix is not None:
                    inputs = self._cached_inputs(inputs, prefix)
                logits = (
                    self.thinker(**inputs, use_cache=False)
                    .logits[0, start - 1 - offset : end - 1 - offset]
                    .double()
                )
                logprobs = logits.log_softmax(-1)
                scores[option] = float(logprobs[torch.arange(len(suffix)), suffix].sum().item())
        return scores

    def hypotheses(self, encoded: Any, prefix: Any = None) -> list[str]:
        prompt = self.asr._build_text_prompt(context="", force_language=None)
        texts: list[str] = []
        with torch.inference_mode():
            for beams in (1, 4):
                inputs = self._inputs(prompt, encoded)
                generation = {}
                if prefix is not None:
                    self.thinker.rope_deltas = prefix["rope_deltas"].clone()
                    cache = deepcopy(prefix["cache"])
                    if beams > 1:
                        cache.batch_repeat_interleave(beams)
                    generation["past_key_values"] = cache
                output = self.asr.model.generate(
                    **inputs,
                    **generation,
                    num_beams=beams,
                    num_return_sequences=beams,
                    max_new_tokens=max(128, math.ceil(len(encoded) / 100) * 128),
                    do_sample=False,
                    length_penalty=1.0,
                )
                decoded = self.processor.batch_decode(
                    output.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                texts.extend(parse_asr_output(raw)[1] for raw in decoded)
        if len(texts) != 5:
            raise ValueError("Expected greedy plus four beam hypotheses")
        return texts

    @staticmethod
    def term_score(term: str, hypotheses: list[str]) -> float:
        return exact_fraction(term, hypotheses)
