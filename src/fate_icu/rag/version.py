"""Index versions and file hashes (Stage 5: "version the index in the run manifest").

The index version is a hash of everything that determines the BM25 and FAISS indexes and
the query embeddings they are searched with: the corpus hash, the BM25 parameters, both
MedCPT encoders at their pinned revisions, and the embedding settings. The query bank adds
its own hash on top (queries, fusion and rerank settings, the cross-encoder revision).
Every downstream run manifest records both.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fate_icu.rag.config import RagConfig

INDEX_SCHEMA_VERSION = 1
INDEX_INFO_FILE = "index.json"


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_payload(corpus_sha256: str, config: RagConfig) -> dict[str, Any]:
    """What the index version covers."""
    return {
        "schema": INDEX_SCHEMA_VERSION,
        "corpus_sha256": corpus_sha256,
        "bm25": config.bm25.model_dump(),
        "query_encoder": config.models.query_encoder.model_dump(),
        "article_encoder": config.models.article_encoder.model_dump(),
        "embedding": {
            "max_length": config.embedding.max_length,
            "precision": "fp16-autocast" if config.embedding.autocast_fp16 else "fp32",
            "pooling": "cls",
            "similarity": "inner_product",
        },
        "faiss": "IndexFlatIP",
    }


def index_version(corpus_sha256: str, config: RagConfig) -> str:
    """16 hex characters identifying the index built from this corpus with this config."""
    return _sha256_json(index_payload(corpus_sha256, config))[:16]


def binary_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (for binary files; ``utils.hashing`` normalises line endings)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def files_sha256(root: str | Path, files: Iterable[Path]) -> dict[str, str]:
    """``{posix path relative to root: sha256}`` for the given files, sorted by path."""
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): binary_sha256(p) for p in sorted(files, key=lambda p: p.as_posix())
    }
