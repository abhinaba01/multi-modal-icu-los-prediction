"""Tiny BERT encoders, cross-encoder and WordPiece tokenizer built offline for the Stage 5 tests.

Random weights (fixed seeds): they test plumbing (pooling, batching, ordering, file
formats), not retrieval quality. In transformers 5, ``BertTokenizer`` takes the vocabulary
as a dict (``vocab_file=`` is silently ignored).
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

WORDS = """
a about acidosis acute admission adults after air an and are arterial as associated at be blood by
care cause causes chest clearance creatinine critical critically day days deep defined delirium dose
early effusion fall first fluid fluids for from function given glasgow guideline high higher hours
hypotension icu ill in injury intensive is it kidney lactate length less light longer low lung mean
mechanical mortality needs norepinephrine of on or organ output oxygen patients pleural pressure
prolonged radiograph raised rise score scale sedation sepsis septic serum shock shorter sofa stage
stay stayed study synthetic the tissue to treatment unit urine vasopressor vasopressors ventilated
ventilation was were when with within
""".split()  # noqa: SIM905 - a word list reads better as text
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PIECES = [
    "##s",
    "##ed",
    "##ing",
    "##ly",
    "##al",
    ".",
    ",",
    ":",
    ";",
    "(",
    ")",
    "/",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
]


def tiny_vocab() -> dict[str, int]:
    tokens = SPECIAL + sorted(set(WORDS)) + PIECES
    return {token: i for i, token in enumerate(tokens)}


def tiny_tokenizer() -> BertTokenizer:
    return BertTokenizer(vocab=tiny_vocab())


def tiny_config(**kw) -> BertConfig:
    return BertConfig(
        vocab_size=len(tiny_vocab()),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        **kw,
    )


def tiny_encoder(seed: int = 0) -> BertModel:
    torch.manual_seed(seed)
    return BertModel(tiny_config()).eval()


def tiny_cross_encoder(seed: int = 1) -> BertForSequenceClassification:
    torch.manual_seed(seed)
    return BertForSequenceClassification(tiny_config(num_labels=1)).eval()


def save_tiny(directory: str | Path, model, tokenizer) -> Path:
    """Save a tiny model and tokenizer like a Hugging Face snapshot (for from_pretrained)."""
    directory = Path(directory)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
