"""Frozen question-conditioned speech head; imported only by the torch engine."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import WhisperTokenizer

TEXT_REPO = "openai/whisper-tiny"
TEXT_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
ALPHABET = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç0123456789 -'"


def transformer_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, norm_first=True
    )


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_whisper = nn.Embedding(51865, 384)
        self.projection = nn.Linear(384, 256, bias=False)
        self.positions = nn.Parameter(torch.empty(128, 256))
        self.transformer = nn.TransformerEncoder(transformer_layer(), num_layers=2)
        self.normalized = nn.LayerNorm(256)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        hidden = self.projection(self.embedding_whisper(ids)) + self.positions[: ids.shape[1]]
        hidden = self.transformer(hidden, src_key_padding_mask=~mask.bool())
        return cast(Tensor, self.normalized(hidden).masked_fill(~mask.bool().unsqueeze(-1), 0.0))


class GraphemeTokenizer:
    def __call__(self, text: str) -> tuple[Tensor, Tensor]:
        vocabulary = {character: index + 4 for index, character in enumerate(ALPHABET)}
        ids = [2] + [
            vocabulary.get(char, 1) for char in unicodedata.normalize("NFC", text).casefold()
        ]
        ids = ids[:63] + [3]
        tokens = torch.tensor([ids], dtype=torch.long)
        return tokens, torch.ones_like(tokens, dtype=torch.bool)


class GraphemeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(ALPHABET) + 4, 256, padding_idx=0)
        self.positions = nn.Parameter(torch.empty(64, 256))
        self.convolution = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(transformer_layer(), num_layers=2)
        self.normalized = nn.LayerNorm(256)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        hidden = self.embedding(ids) + self.positions[: ids.shape[1]]
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        convolved = self.convolution(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = (hidden + F.gelu(convolved)).masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden = self.transformer(hidden, src_key_padding_mask=~mask)
        return cast(Tensor, self.normalized(hidden).masked_fill(~mask.unsqueeze(-1), 0.0))


class ConditionedAudioLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = transformer_layer()
        self.film = nn.Linear(256, 512)

    def forward(self, audio: Tensor, question: Tensor, mask: Tensor) -> Tensor:
        audio = self.transformer(audio, src_key_padding_mask=~mask.bool())
        gamma, beta = self.film(question).chunk(2, dim=-1)
        return cast(Tensor, audio * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1))


class MarimbaHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection_audio = nn.Linear(896, 256)
        self.layers_audio = nn.ModuleList([ConditionedAudioLayer() for _ in range(2)])
        self.normalized_audio = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.score = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
        self.frame_score = nn.Linear(256, 3)
        self.speaker_frame_score = nn.Linear(256, 4)
        self.embedding_side = nn.Embedding(4, 256)

    @staticmethod
    def pool(hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def forward(
        self,
        features: Tensor,
        mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
        option: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        question_pool = self.pool(question, question_mask)
        memory = self.projection_audio(features)
        for layer in self.layers_audio:
            memory = layer(memory, question_pool, mask)
        memory = self.normalized_audio(memory)
        context, _ = self.attention(
            option, memory, memory, key_padding_mask=~mask.bool(), need_weights=False
        )
        return cast(Tensor, self.score(self.pool(option + context, option_mask)).reshape(-1))


class SpeechModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text = TextEncoder()
        self.head = MarimbaHead()
        self.text_options = GraphemeEncoder()


class CachedQuestionTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.tokens = lru_cache(maxsize=128)(self._tokens)

    def _tokens(self, text: str) -> tuple[list[int], list[int]]:
        encoded = self.tokenizer(text, truncation=True, max_length=64)
        return encoded["input_ids"], encoded["attention_mask"]

    def __call__(self, texts: Sequence[str]) -> Any:
        values = [self.tokens(text) for text in texts]
        return self.tokenizer.pad(
            {
                "input_ids": [value[0] for value in values],
                "attention_mask": [value[1] for value in values],
            },
            padding=True,
            return_tensors="pt",
        )


class SpeechReadout:
    def __init__(self, paths: dict[str, Path], device: str) -> None:
        config = json.loads(paths["speech_head/config.json"].read_text())
        if config != {
            "format_version": 1,
            "layer": 9,
            "frames": 100,
            "dimension_audio": 896,
            "dimension": 256,
            "heads": 4,
            "text_layers": 2,
            "tail_pool_seconds": 0,
            "positive_option": "yes",
            "tokenizer_repo": TEXT_REPO,
            "tokenizer_revision": TEXT_REVISION,
        }:
            raise ValueError("Unsupported speech head configuration")
        self.device = device
        self.model = SpeechModel()
        self.model.load_state_dict(load_file(paths["speech_head/model.safetensors"]), strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.tokenizer = CachedQuestionTokenizer(
            WhisperTokenizer.from_pretrained(TEXT_REPO, revision=TEXT_REVISION)
        )
        # The head was validated with the "sim" option.
        ids, mask = GraphemeTokenizer()("sim")
        with torch.inference_mode():
            self.option = self.model.text_options(ids.to(device), mask.to(device))
        self.option_mask = mask.to(device)
        self.question_features = self._question_features

    def _question_features(self, text: str) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer([text])
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].bool().to(self.device)
        return self.model.text(ids, mask), mask

    def score(self, native: Tensor, sample_mask: np.ndarray[Any, Any], text: str) -> float:
        frames = (
            F.interpolate(native.T.unsqueeze(0), size=100, mode="linear", align_corners=False)
            .transpose(1, 2)
            .to(torch.float16)
            .float()
            .to(self.device)
        )
        frame_mask = np.asarray(sample_mask.reshape(100, -1).any(axis=1), dtype=np.bool_)
        mask = torch.from_numpy(frame_mask[None]).to(self.device)
        with torch.inference_mode():
            question, question_mask = self.question_features(text)
            logit = self.model.head(
                frames, mask, question, question_mask, self.option, self.option_mask
            )
        return float(logit.item())
