"""Optional torch engine with one shared audio-tower pass per window."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output
from safetensors.numpy import load_file

from ._head import SpeechReadout
from .audio import AudioArray, Mask
from .language import LANGUAGES
from .lexical import exact_fraction
from .readout import pool_layer, predict_readout
from .runtime import RuntimeInfo
from .weights import ENCODER_REPO, ENCODER_REVISION


class QwenEngine:
    def __init__(self, runtime: RuntimeInfo, paths: dict[str, Path]) -> None:
        self.asr = Qwen3ASRModel.from_pretrained(
            snapshot_download(ENCODER_REPO, revision=ENCODER_REVISION),
            dtype=getattr(torch, runtime.dtype),
            device_map=runtime.device,
            attn_implementation="sdpa",
        )
        self.asr.model.eval().requires_grad_(False)
        self.processor = self.asr.processor
        self.processor.tokenizer.padding_side = "left"
        self.thinker = self.asr.model.thinker
        self.taps: dict[int, Any] = {}
        self.raw_taps: dict[int, Any] = {}
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
        self.raw_taps[layer] = hidden.detach()
        if layer == 18:
            self.taps[18] = hidden.detach().to(torch.float16).float().cpu().numpy()

    def encode(self, window: AudioArray) -> Any:
        features = self.processor.feature_extractor(
            [window],
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = features["input_features"].to(self.asr.model.device, self.asr.model.dtype)
        attention_mask = features["attention_mask"].to(self.asr.model.device)
        self.taps.clear()
        self.raw_taps.clear()
        with torch.inference_mode():
            encoded = self.thinker.get_audio_features(inputs, feature_attention_mask=attention_mask)
        return encoded

    def _inputs(self, text: str, encoded: Any) -> dict[str, Any]:
        expanded = self.processor.replace_multimodal_special_tokens([text], iter([len(encoded)]))
        batch = self.processor.tokenizer(expanded, return_tensors="pt", padding=True)
        batch = batch.to(self.asr.model.device)
        embeddings = self.thinker.get_input_embeddings()(batch["input_ids"])
        audio_mask = self.thinker.get_placeholder_mask(batch["input_ids"], inputs_embeds=embeddings)
        embeddings = embeddings.masked_scatter(audio_mask, encoded.to(embeddings.dtype))
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "inputs_embeds": embeddings,
        }

    def turn_end(self, mask: Mask) -> tuple[float, float]:
        return predict_readout(self.readout, pool_layer(self.taps[18], mask))

    def head_score(self, mask: Mask, text: str) -> float:
        if self.head is None:
            raise ValueError("No speech head weights are installed")
        return self.head.score(self.raw_taps[9], mask, text)

    def language_scores(self, encoded: Any, options: Sequence[str]) -> dict[str, float]:
        base = self.asr._build_text_prompt(context="", force_language=None) + "language"
        tokenizer = self.processor.tokenizer
        prefix = tokenizer.encode(base, add_special_tokens=False)
        scores = {}
        with torch.inference_mode():
            for option in options:
                if option not in LANGUAGES:
                    raise ValueError(
                        f"Unsupported language code {option!r}; supported: {', '.join(LANGUAGES)}"
                    )
                text = base + " " + LANGUAGES[option]
                full = tokenizer.encode(text, add_special_tokens=False)
                if full[: len(prefix)] != prefix:
                    raise ValueError("Native language prompt has an unstable token prefix")
                suffix = full[len(prefix) :]
                inputs = self._inputs(text, encoded)
                end = inputs["input_ids"].shape[1]
                start = end - len(suffix)
                if inputs["input_ids"][0, start:end].tolist() != suffix:
                    raise ValueError("Native language suffix token positions do not match")
                logits = (
                    self.thinker(**inputs, use_cache=False).logits[0, start - 1 : end - 1].double()
                )
                logprobs = logits.log_softmax(-1)
                scores[option] = float(logprobs[torch.arange(len(suffix)), suffix].sum().item())
        return scores

    def hypotheses(self, encoded: Any) -> list[str]:
        prompt = self.asr._build_text_prompt(context="", force_language=None)
        texts: list[str] = []
        with torch.inference_mode():
            for beams in (1, 4):
                inputs = self._inputs(prompt, encoded)
                output = self.asr.model.generate(
                    **inputs,
                    num_beams=beams,
                    num_return_sequences=beams,
                    max_new_tokens=128,
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
