"""Stage 5 fusion: reciprocal-rank fusion and cross-encoder reranking (hand-computed)."""

from __future__ import annotations

import numpy as np
import pytest

from fate_icu.rag.fusion import rerank, rrf


def test_rrf_hand_computed():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [row for row, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1][1] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2][1] == pytest.approx(1 / 62)


def test_rrf_ties_by_row_and_edges():
    assert [row for row, _ in rrf([[7, 2], [2, 7]])] == [2, 7]  # equal sums: the lower row first
    assert rrf([]) == [] and rrf([[], []]) == []
    assert rrf([[5]], k=0) == [(5, 1.0)]
    with pytest.raises(ValueError, match="twice"):
        rrf([[1, 1]])


class FixedCross:
    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def score(self, query, texts):
        self.seen.append(list(texts))
        return np.array([self.scores[t] for t in texts], dtype=np.float32)


def test_rerank_depth_order_and_ties():
    cross = FixedCross({"a": 0.5, "b": 2.0, "c": 2.0, "d": 9.0})
    assert rerank("q", [10, 11, 12, 13], ["a", "b", "c", "d"], cross, depth=3) == [
        (11, 2.0),
        (12, 2.0),
        (10, 0.5),
    ]
    assert cross.seen == [["a", "b", "c"]]  # only the first depth candidates are scored
    assert rerank("q", [], [], cross, depth=5) == []
    with pytest.raises(ValueError, match="texts"):
        rerank("q", [1, 2], ["a"], cross, depth=2)
