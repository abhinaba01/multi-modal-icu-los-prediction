"""Passage embeddings with the MedCPT article encoder (Stage 5, run by ``gpu_step``).

The full corpus is embedded on a Kaggle or Colab GPU; the smoke corpus on the laptop CPU.
Shards of ``shard_size`` passages are saved as they finish, so a rerun resumes. The corpus
is re-hashed first and must match the index's ``gpu_spec.json``.

This module imports only what the GPU machine needs (the corpus schema, the encoder and the
embedding writer): no BM25, PDF or config packages.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fate_icu.rag.corpus import Passage, corpus_sha256
from fate_icu.rag.dense import embed_corpus
from fate_icu.rag.encoders import ArticleEncoder, ModelRef

DENSE_DIR = "dense"
# A CPU embeds roughly 1-20 passages a second: above this, use a GPU (A4: ask before > 10 min).
CPU_PASSAGE_LIMIT = 2_000


def embedding_model_info(embedding: dict[str, Any]) -> dict[str, Any]:
    """What the embeddings record about the model; checked again when they are installed."""
    return {
        "repo": embedding["model"]["repo"],
        "revision": embedding["model"]["revision"],
        "max_length": embedding["max_length"],
        "precision": "fp16-autocast" if embedding["autocast_fp16"] else "fp32",
    }


def check_corpus(passages: list[Passage], spec: dict[str, Any]) -> str:
    """The corpus hash, after checking it against the spec (fails loudly on any difference)."""
    digest = corpus_sha256(passages)
    if digest != spec["corpus_sha256"] or len(passages) != spec["n_passages"]:
        raise RuntimeError(
            f"the corpus file ({len(passages)} passages, hash {digest[:12]}) does not match gpu_spec.json "
            f"({spec['n_passages']} passages, hash {spec['corpus_sha256'][:12]})"
        )
    return digest


def embed(
    passages: list[Passage],
    spec: dict[str, Any],
    out_dir: Path,
    *,
    device: str,
    encoder: Any = None,
    log=print,
) -> dict[str, Any]:
    """Embed the passages shard by shard into ``out_dir`` (resumable)."""
    digest = check_corpus(passages, spec)
    embedding = spec["embedding"]
    if encoder is None:
        encoder = ArticleEncoder.from_ref(
            ModelRef(**embedding["model"]),
            device=device,
            max_length=embedding["max_length"],
            batch_size=embedding["batch_size"],
            autocast_fp16=bool(embedding["autocast_fp16"]) and device == "cuda",
        )
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        log(f"embedded {done:,} / {total:,} passages ({time.monotonic() - started:,.0f} s)")

    info = embed_corpus(
        [p.encoder_pair() for p in passages],
        encoder,
        out_dir,
        corpus_sha256=digest,
        model=embedding_model_info(embedding),
        shard_size=embedding["shard_size"],
        progress=progress,
    )
    return {**info.model_dump(), "device": device, "seconds": round(time.monotonic() - started, 1)}
