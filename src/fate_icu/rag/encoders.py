"""MedCPT query encoder, article encoder and cross-encoder at pinned revisions (Stage 5).

As in the MedCPT model cards: the embedding is the ``[CLS]`` vector of the last hidden
layer, with no normalisation (retrieval uses raw inner products); the article encoder reads
``(title, text)`` pairs (max 512 tokens), the query encoder a query (max 64 tokens), and the
cross-encoder ``[query, "Title. Text"]`` pairs, returning one logit per pair. With
transformers 5.22.2 all three reproduce their model-card outputs (checked offline).

Inputs are batched in order of length (longest first, ties by input position), then put
back in input order, so the output depends only on the inputs, not on the batch size beyond
float rounding.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ModelRef(BaseModel):
    """A Hugging Face model at an exact commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @property
    def short(self) -> str:
        return f"{self.repo}@{self.revision[:8]}"


MEDCPT_QUERY = ModelRef(repo="ncbi/MedCPT-Query-Encoder", revision="d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc")
MEDCPT_ARTICLE = ModelRef(
    repo="ncbi/MedCPT-Article-Encoder", revision="d05a736da4bb84ee4057b7f7999485be6ed85465"
)
MEDCPT_CROSS = ModelRef(repo="ncbi/MedCPT-Cross-Encoder", revision="71caf65d4927987813984f54c284405a13fcca49")


def load_pretrained(
    ref: ModelRef, kind: Literal["encoder", "cross"], *, device: str = "cpu", local_files_only: bool = False
) -> tuple[Any, Any]:
    """``(model, tokenizer)`` at ``ref.revision``, in eval mode on ``device``."""
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # weight-loading bars would flood the console output
    tokenizer = AutoTokenizer.from_pretrained(
        ref.repo, revision=ref.revision, local_files_only=local_files_only
    )
    model_class = AutoModel if kind == "encoder" else AutoModelForSequenceClassification
    model = model_class.from_pretrained(ref.repo, revision=ref.revision, local_files_only=local_files_only)
    return model.eval().to(device), tokenizer


class _Batched:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cpu",
        max_length: int,
        batch_size: int = 32,
        name: str = "",
        ref: ModelRef | None = None,
        autocast_fp16: bool = False,
    ):
        if autocast_fp16 and device != "cuda":
            raise ValueError("fp16 autocast needs a CUDA device")
        self.model, self.tokenizer = model, tokenizer
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.name, self.ref, self.autocast_fp16 = name or (ref.short if ref else ""), ref, autocast_fp16

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": self.ref.repo if self.ref else None,
            "revision": self.ref.revision if self.ref else None,
            "max_length": self.max_length,
            "precision": "fp16-autocast" if self.autocast_fp16 else "fp32",
            "device": self.device,
        }

    @classmethod
    def from_ref(cls, ref: ModelRef, *, device: str = "cpu", local_files_only: bool = False, **kw: Any):
        model, tokenizer = load_pretrained(ref, cls._kind, device=device, local_files_only=local_files_only)
        return cls(model, tokenizer, device=device, ref=ref, **kw)

    _kind: Literal["encoder", "cross"] = "encoder"

    def _run(self, inputs: list, forward: Callable[[dict], Any]) -> np.ndarray:
        import torch

        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)
        lengths = [
            len(ids)
            for ids in self.tokenizer(inputs, truncation=True, max_length=self.max_length, verbose=False)[
                "input_ids"
            ]
        ]
        order = sorted(range(len(inputs)), key=lambda i: (-lengths[i], i))
        outputs: list[np.ndarray | None] = [None] * len(inputs)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast_fp16
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [inputs[i] for i in batch],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                values = forward(encoded).float().cpu().numpy()
                for position, i in enumerate(batch):
                    outputs[i] = values[position]
        return np.stack(outputs).astype(np.float32, copy=False)

    def _cls(self, encoded: dict) -> Any:
        return self.model(**encoded).last_hidden_state[:, 0, :]


class QueryEncoder(_Batched):
    """MedCPT query encoder: ``encode(texts)`` -> float32 ``[n, 768]``."""

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 64, **kw: Any):
        super().__init__(model, tokenizer, max_length=max_length, **kw)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(list(texts), self._cls)


class ArticleEncoder(_Batched):
    """MedCPT article encoder: ``encode([(title, text), ...])`` -> float32 ``[n, 768]``."""

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 512, **kw: Any):
        super().__init__(model, tokenizer, max_length=max_length, **kw)

    def encode(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return self._run([[title, text] for title, text in pairs], self._cls)


class CrossEncoder(_Batched):
    """MedCPT cross-encoder: ``score(query, texts)`` -> one float32 logit per text."""

    _kind = "cross"

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 512, **kw: Any):
        super().__init__(model, tokenizer, max_length=max_length, **kw)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        logits = self._run([[query, text] for text in texts], lambda encoded: self.model(**encoded).logits)
        return logits[:, 0].astype(np.float32)
