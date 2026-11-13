"""Reciprocal-rank fusion and cross-encoder reranking (Stage 5).

RRF (Cormack et al., 2009): a passage's score is the sum over the input rankings of
1 / (k + rank), rank starting at 1, with k = 60; a passage absent from a ranking adds
nothing. Ties are broken by corpus row, here and after reranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fused ``(row, score)`` sorted by (-score, row).

    Raises:
        ValueError: A ranking lists the same row twice.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking lists the same passage twice")
        for rank, row in enumerate(ranking, start=1):
            scores[int(row)] = scores.get(int(row), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def rerank(
    query: str, candidates: Sequence[int], texts: Sequence[str], cross_encoder: Any, *, depth: int
) -> list[tuple[int, float]]:
    """The first ``depth`` candidates rescored by ``cross_encoder.score``, sorted by (-score, row)."""
    if len(candidates) != len(texts):
        raise ValueError(f"{len(candidates)} candidates but {len(texts)} texts")
    chosen = list(candidates[:depth])
    if not chosen:
        return []
    scores = cross_encoder.score(query, list(texts[:depth]))
    return sorted(
        ((int(c), float(s)) for c, s in zip(chosen, scores, strict=True)), key=lambda hit: (-hit[1], hit[0])
    )
