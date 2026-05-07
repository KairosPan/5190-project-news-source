"""
model.py - Project B: News Source Classification (1=FoxNews, 0=NBC)

Single `microsoft/deberta-v3-base` classifier. The evaluator instantiates
`Model(weights_path="__no_weights__.pth")`, loads `model.pt` externally, and
then calls `predict(batch)`. A real `weights_path` is also supported for local
smoke tests and direct reuse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 2
MAX_LENGTH = 96
_SENTINEL_WEIGHTS = {"", "__no_weights__.pth", None}


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def _load_matching(target: nn.Module, state_dict: Mapping[str, Any]) -> int:
    target_state = target.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in target_state
        and isinstance(value, torch.Tensor)
        and target_state[key].shape == value.shape
    }
    if filtered:
        target.load_state_dict(filtered, strict=False)
    return len(filtered)


class Model(nn.Module):
    def __init__(self, weights_path: Optional[str] = None) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="single_label_classification",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_config(self.config)
        self.device = _best_device()
        self.to(self.device)
        self._maybe_load_weights(weights_path)
        self.eval()

    def _maybe_load_weights(self, weights_path: Optional[str]) -> None:
        if weights_path in _SENTINEL_WEIGHTS:
            return
        path = Path(str(weights_path)).expanduser()
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu")
        raw_state = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if not isinstance(raw_state, Mapping):
            raise RuntimeError(
                "Checkpoint must be a state_dict or a dict with a state_dict key."
            )
        state_dict = _normalize_state_dict(raw_state)
        loaded = _load_matching(self.model, state_dict) + _load_matching(
            self, state_dict
        )
        if loaded == 0:
            raise RuntimeError("No checkpoint tensors matched the DeBERTa classifier.")
        self.to(self.device)

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def _encode(self, batch: Iterable[Any]) -> dict[str, torch.Tensor]:
        texts = [str(x) if x is not None else "" for x in batch]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = list(batch)
        if not texts:
            return []
        enc = self._encode(texts)
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        logits = self.model(**enc).logits
        return logits.argmax(dim=-1).detach().cpu().tolist()

    def forward(self, batch: Iterable[Any]) -> torch.Tensor:
        enc = self._encode(batch)
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        return self.model(**enc).logits


def get_model() -> Model:
    return Model()
