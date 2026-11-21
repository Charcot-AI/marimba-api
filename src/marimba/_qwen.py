"""Optional torch engine with one shared audio-tower pass per window."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Iterator, Sequence
from contextvars import ContextVar
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

from ._diarization import TASKS, DiarizationReadout, task_features
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
from .windows import DeviceFeatures


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


# The audio-tower layer each window readout reads.
_TASK_LAYERS = {"voice_activity": 9, "turn_end": 18, **dict.fromkeys(TASKS, 12)}


class _EncodedBatch:
    """Activations of one batched pass, left on the device until a readout asks for them."""

    def __init__(self, captures: dict[int, Any], audio: Any, head: SpeechReadout | None) -> None:
        self.captures = captures
        self._audio = audio
        self._head = head
        self._arrays: dict[int, Any] = {}
        self._scores: dict[tuple[str, bytes], list[float]] = {}

    def layer(self, index: int) -> Any:
        if index not in self._arrays:
            values = self.captures[index].float().cpu().numpy()
            values.flags.writeable = False
            self._arrays[index] = values
        return self._arrays[index]

    def audio(self) -> Any:
        if self._audio is None:
            raise RuntimeError("This batch was encoded without decoder features")
        if not isinstance(self._audio, np.ndarray):
            self._audio = self._audio.float().cpu().numpy()
            self._audio.flags.writeable = False
        return self._audio

    def head_score(self, row: int, mask: Mask, text: str) -> float:
        if self._head is None:
            raise ValueError("No speech head weights are installed")
        key = (text, np.packbits(mask.reshape(100, -1).any(axis=1)).tobytes())
        if key not in self._scores:
            # Rows of a batch ask the same question over the same tail: one head pass for all.
            self._scores[key] = self._head.score_batch(self.captures[9], mask, text)
        return self._scores[key][row]


class _RowFeatures(DeviceFeatures):
    """Captured layers of one window of a batch, read as host arrays on demand."""

    def __init__(self, batch: _EncodedBatch, row: int) -> None:
        self.batch = batch
        self.row = row

    def __getitem__(self, layer: int) -> Any:
        if layer not in self.batch.captures:
            raise KeyError(layer)
        return self.batch.layer(layer)[self.row]

    def __iter__(self) -> Iterator[int]:
        return iter(self.batch.captures)

    def __len__(self) -> int:
        return len(self.batch.captures)


class _RowAudio:
    """Decoder features of one window of a batch, copied to the host only when read."""

    def __init__(self, batch: _EncodedBatch, row: int, steps: int) -> None:
        self._batch = batch
        self._row = row
        self._steps = steps

    def __len__(self) -> int:
        return self._steps

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        values = self._batch.audio()[self._row]
        return values.astype(dtype) if dtype is not None else values.copy()


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
        self.diarization_readouts = {
            task: DiarizationReadout(task, load_file(paths[f"diarization/{task}.safetensors"]))
            for task in TASKS
            if f"diarization/{task}.safetensors" in paths
        }
        if self.diarization_readouts:
            layers = (*layers, 12)
        self._layers = layers
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
            if layer == 12 and hidden.ndim == 3:
                if hidden.shape[0] != 1:
                    raise ValueError("Diarization requires one window per encoder pass")
                hidden = hidden[0]
            if layer in (12, 18):
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

    def _log_mel(self, windows: Sequence[AudioArray]) -> Any:
        """Log-mel of equal-length windows on the model device: the arithmetic of the Whisper
        feature extractor, without its host round trip."""
        extractor = self.processor.feature_extractor
        device = self.asr.model.device
        waveform = torch.from_numpy(np.stack(windows)).to(device, torch.float32)
        window = torch.hann_window(extractor.n_fft, device=device)
        stft = torch.stft(
            waveform, extractor.n_fft, extractor.hop_length, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        filters = torch.from_numpy(extractor.mel_filters).to(device, torch.float32)
        spectrum = torch.clamp(filters.T @ magnitudes, min=1e-10).log10()
        spectrum = torch.maximum(spectrum, spectrum.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (spectrum + 4.0) / 4.0

    def encode_batch(
        self, windows: Sequence[AudioArray], tasks: Collection[str] | None = None
    ) -> list[Any]:
        """One audio-tower forward for several eight-second windows.

        With ``tasks``, the pass stops at the last layer those window readouts read and
        skips the decoder features.
        """
        tower = self.thinker.audio_tower
        extractor = self.processor.feature_extractor
        samples, chunk = len(windows[0]) if windows else 0, tower.n_window * 2
        frames = samples // extractor.hop_length
        # The sequential pass attends over a whole window under SDPA; other layouts keep it.
        if (
            len(windows) < 2
            or tower.config._attn_implementation != "sdpa"
            or any(len(window) != samples for window in windows)
            or samples % extractor.hop_length
            or frames % chunk
            or getattr(extractor, "dither", 0.0) != 0.0
        ):
            return [self.encode(window) for window in windows]
        # Window readouts name the layers they read; any other question needs the full pass.
        known = bool(tasks) and all(task in _TASK_LAYERS for task in tasks or ())
        wanted = set(self._layers)
        if known:
            wanted &= {_TASK_LAYERS[task] for task in tasks or ()}
        last = max(wanted) if known and wanted else len(tower.layers)
        with torch.inference_mode():
            mel = self._log_mel(windows).to(self.asr.model.dtype)
            batch, bins = mel.shape[:2]
            chunks = mel.unflatten(2, (-1, chunk)).permute(0, 2, 1, 3).reshape(-1, 1, bins, chunk)
            if mel.is_cuda:
                # The convolutions are most of the pass; NHWC is their faster cuDNN layout.
                if not getattr(self, "_channels_last", False):
                    for convolution in (tower.conv2d1, tower.conv2d2, tower.conv2d3):
                        convolution.to(memory_format=torch.channels_last)
                    self._channels_last = True
                chunks = chunks.contiguous(memory_format=torch.channels_last)
            embeds = []
            for part in chunks.split(tower.conv_chunksize, dim=0):
                part = F.gelu(tower.conv2d1(part))
                part = F.gelu(tower.conv2d2(part))
                embeds.append(F.gelu(tower.conv2d3(part)))
            embed = torch.cat(embeds, dim=0)
            count, channels, height, steps = embed.shape
            embed = tower.conv_out(
                embed.permute(0, 3, 1, 2).contiguous().view(count, steps, channels * height)
            )
            positions = tower.positional_embedding.positional_embedding[:steps]
            hidden = (embed + positions.to(embed.dtype)).reshape(batch, -1, embed.shape[-1])
            captures: dict[int, Any] = {}
            for index, layer in enumerate(tower.layers[:last], 1):
                hidden = self._batched_layer(layer, hidden)
                if index in wanted:
                    captures[index] = hidden.to(torch.float16) if index in (12, 18) else hidden
            audio = None
            if not known:
                audio = tower.proj2(tower.act(tower.proj1(tower.ln_post(hidden))))
        encoded = _EncodedBatch(captures, audio, self.head)
        length = 0 if audio is None else audio.shape[1]
        return [
            {
                "features": _RowFeatures(encoded, row),
                "audio_features": None if audio is None else _RowAudio(encoded, row, length),
            }
            for row in range(batch)
        ]

    @staticmethod
    def _batched_layer(layer: Any, hidden: Any) -> Any:
        attention = layer.self_attn
        batch, steps, _ = hidden.shape
        query, key, value = (
            projection(layer.self_attn_layer_norm(hidden))
            .view(batch, steps, attention.num_heads, -1)
            .transpose(1, 2)
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        )
        attended = F.scaled_dot_product_attention(query, key, value, scale=attention.scaling)
        hidden = hidden + attention.out_proj(attended.transpose(1, 2).reshape(batch, steps, -1))
        hidden = hidden + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden))))
        if hidden.dtype == torch.float16:
            limit = torch.finfo(hidden.dtype).max - 1000
            hidden = torch.clamp(hidden, min=-limit, max=limit)
        return hidden

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

    def diarization(self, features: Any, tasks: set[str]) -> dict[str, Any]:
        if 12 not in features:
            raise RuntimeError("Encoded window lacks layer 12 for diarization")
        return {
            task: self.diarization_readouts[task].predict(task_features(task, features[12])[None])[
                0
            ]
            for task in tasks
        }

    def head_score(self, features: Any, mask: Mask, text: str) -> float:
        if isinstance(features, _RowFeatures):
            return features.batch.head_score(features.row, mask, text)
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
                logits = self.thinker(**inputs, use_cache=False).logits[
                    0, start - 1 - offset : end - 1 - offset
                ]
                if logits.device.type == "mps":
                    logits = logits.cpu()
                logits = logits.double()
                logprobs = logits.log_softmax(-1)
                scores[option] = float(logprobs[torch.arange(len(suffix)), suffix].sum().item())
        return scores

    def _generate(self, encoded: Any, prefix: Any, beams: int) -> list[str]:
        prompt = self.asr._build_text_prompt(context="", force_language=None)
        with torch.inference_mode():
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
        return [parse_asr_output(raw)[1] for raw in decoded]

    def transcript(self, encoded: Any, prefix: Any = None) -> str:
        texts = self._generate(encoded, prefix, beams=1)
        if len(texts) != 1:
            raise ValueError("Expected one greedy transcript")
        return texts[0]

    def hypotheses(self, encoded: Any, prefix: Any = None) -> list[str]:
        texts = self._generate(encoded, prefix, beams=1) + self._generate(encoded, prefix, beams=4)
        if len(texts) != 5:
            raise ValueError("Expected greedy plus four beam hypotheses")
        return texts

    @staticmethod
    def term_score(term: str, hypotheses: list[str]) -> float:
        return exact_fraction(term, hypotheses)
