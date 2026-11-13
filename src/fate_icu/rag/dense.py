"""Passage embeddings in resumable shards, and the exact FAISS inner-product index (Stage 5).

:func:`embed_corpus` writes ``shards/shard-<i>.npy`` as each block of ``shard_size``
passages finishes (a rerun reuses finished shards), then ``embeddings.npy`` (float32,
corpus order) and ``embeddings.json`` (corpus hash, model, shape, SHA-256).

:class:`DenseIndex` is a flat (exact) ``IndexFlatIP``. FAISS returns tied scores in an
arbitrary order, so every result is re-sorted by (-score, row); when a tie could straddle
the cut-off, that query is scored exactly against all vectors.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from fate_icu.rag.version import binary_sha256

EMBEDDINGS_FILE = "embeddings.npy"
EMBEDDINGS_INFO_FILE = "embeddings.json"
SHARDS_DIR = "shards"
TIE_MARGIN = 16  # extra FAISS results fetched to see ties at the cut-off


class EmbeddingsInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_sha256: str
    model: dict[str, Any]
    n: int
    dim: int
    shard_size: int
    sha256: str


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def embed_corpus(
    pairs: Sequence[tuple[str, str]],
    encoder: Any,
    out_dir: str | Path,
    *,
    corpus_sha256: str,
    model: dict[str, Any],
    shard_size: int = 50_000,
    progress: Callable[[int, int], None] | None = None,
) -> EmbeddingsInfo:
    """Embed ``pairs`` (corpus order) with ``encoder.encode`` into ``out_dir``, resumably."""
    out_dir = Path(out_dir)
    shards = out_dir / SHARDS_DIR
    shards.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    if n == 0:
        raise ValueError("no passages to embed")
    paths, dim = [], None
    for s in range(math.ceil(n / shard_size)):
        start, end = s * shard_size, min(n, (s + 1) * shard_size)
        npy, meta = shards / f"shard-{s:05d}.npy", shards / f"shard-{s:05d}.json"
        expected = {"rows": [start, end], "corpus_sha256": corpus_sha256, "model": model}
        reuse = False
        if npy.exists() and meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            shape = np.load(npy, mmap_mode="r").shape
            reuse = {k: saved.get(k) for k in expected} == expected and shape == (
                end - start,
                saved.get("dim"),
            )
        if not reuse:
            vectors = np.ascontiguousarray(encoder.encode(pairs[start:end]), dtype=np.float32)
            if vectors.shape[0] != end - start:
                raise RuntimeError(
                    f"the encoder returned {vectors.shape[0]} vectors for {end - start} passages"
                )
            _save_npy(npy, vectors)
            _write_json(meta, {**expected, "dim": int(vectors.shape[1])})
        shard_dim = json.loads(meta.read_text(encoding="utf-8"))["dim"]
        if dim is not None and shard_dim != dim:
            raise RuntimeError(f"shard {s} has dimension {shard_dim}, earlier shards {dim}")
        dim = shard_dim
        paths.append(npy)
        if progress:
            progress(end, n)
    final = out_dir / EMBEDDINGS_FILE
    tmp = out_dir / (EMBEDDINGS_FILE + ".tmp")
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, dim))
    row = 0
    for path in paths:
        block = np.load(path)
        array[row : row + len(block)] = block
        row += len(block)
    array.flush()
    del array
    os.replace(tmp, final)
    info = EmbeddingsInfo(
        corpus_sha256=corpus_sha256,
        model=model,
        n=n,
        dim=dim,
        shard_size=shard_size,
        sha256=binary_sha256(final),
    )
    _write_json(out_dir / EMBEDDINGS_INFO_FILE, info.model_dump())
    return info


def load_embeddings(
    out_dir: str | Path,
    *,
    expect_corpus_sha256: str,
    expect_model: dict[str, Any] | None = None,
    mmap: bool = False,
) -> np.ndarray:
    """The embeddings of ``out_dir``, after checking them against ``embeddings.json``.

    Raises:
        RuntimeError: Another corpus or model, a different shape, or (unless ``mmap``) a
            file whose SHA-256 differs from the record.
    """
    out_dir = Path(out_dir)
    info = EmbeddingsInfo.model_validate_json((out_dir / EMBEDDINGS_INFO_FILE).read_text(encoding="utf-8"))
    if info.corpus_sha256 != expect_corpus_sha256:
        raise RuntimeError(
            f"{out_dir}: embeddings of corpus {info.corpus_sha256[:12]}, not {expect_corpus_sha256[:12]}"
        )
    if expect_model is not None and info.model != expect_model:
        raise RuntimeError(f"{out_dir}: embeddings made with {info.model}, expected {expect_model}")
    path = out_dir / EMBEDDINGS_FILE
    array = np.load(path, mmap_mode="r" if mmap else None)
    if array.shape != (info.n, info.dim) or array.dtype != np.float32:
        raise RuntimeError(
            f"{path}: shape {array.shape} {array.dtype}, expected ({info.n}, {info.dim}) float32"
        )
    if not mmap and binary_sha256(path) != info.sha256:
        raise RuntimeError(f"{path} differs from the SHA-256 in {EMBEDDINGS_INFO_FILE}")
    return array


class DenseIndex:
    """Exact inner-product search over float32 vectors (FAISS ``IndexFlatIP``)."""

    def __init__(self, index: Any):
        self.index = index
        self._vectors: np.ndarray | None = None

    @classmethod
    def build(cls, embeddings: np.ndarray) -> DenseIndex:
        import faiss

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError(f"need a non-empty [n, d] array, got shape {vectors.shape}")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> DenseIndex:
        import faiss

        return cls(faiss.read_index(str(path)))

    def _exact(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._vectors is None:
            self._vectors = self.index.reconstruct_n(0, self.ntotal)
        scores = self._vectors @ vector
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        """Top ``k`` rows per query by (-inner product, row); ``k`` is capped at ``ntotal``."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        if queries.shape[1] != self.dim:
            raise ValueError(f"queries have dimension {queries.shape[1]}, the index {self.dim}")
        k = min(k, self.ntotal)
        if k <= 0:
            return [[] for _ in queries]
        fetch = min(self.ntotal, k + TIE_MARGIN)
        scores, rows = self.index.search(queries, fetch)
        results = []
        for q in range(len(queries)):
            hits = sorted(
                ((int(i), float(s)) for i, s in zip(rows[q], scores[q], strict=True) if i >= 0),
                key=lambda hit: (-hit[1], hit[0]),
            )
            if fetch < self.ntotal and hits[k - 1][1] == hits[-1][1]:
                hits = self._exact(queries[q], k)  # a tie may straddle the cut-off
            results.append(hits[:k])
        return results
