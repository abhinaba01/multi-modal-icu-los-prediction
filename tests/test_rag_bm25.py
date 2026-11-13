"""Stage 5 BM25: stemming, stopwords, zero scores, tie-breaking, save and load."""

from __future__ import annotations

import pytest

from fate_icu.rag.bm25 import BM25Index, BM25Params

TEXTS = [
    "Patients were mechanically ventilated for five days.",
    "Lactate predicts mortality in septic shock.",
    "The weather report for the coast.",
    "Ventilation weaning protocols shorten the intensive care stay.",
    "Lactate predicts mortality in septic shock.",  # a duplicate of row 1: tied scores
    "Serum creatinine rises in acute kidney injury.",
]


@pytest.fixture(scope="module")
def index():
    return BM25Index.build(TEXTS)


def test_exact_terms_and_stemming(index):
    assert index.tokenize_query("Ventilated patients") == ["ventil", "patient"]
    hits = index.search("ventilated", 10)
    assert {row for row, _ in hits} == {0, 3}  # "ventilated" and "ventilation" share a stem
    assert index.search("creatinine kidney", 3)[0][0] == 5


def test_stopwords_unknown_terms_and_zero_scores(index):
    assert index.search("the and of for", 5) == []
    assert index.search("zzzz", 5) == []
    lactate = index.search("lactate", 10)
    assert {row for row, _ in lactate} == {1, 4} and all(score > 0 for _, score in lactate)
    assert index.scores("zzzz").tolist() == [0.0] * len(TEXTS)


def test_ties_are_broken_by_row(index):
    hits = index.search("lactate septic shock", 10)
    assert hits[0][1] == hits[1][1] and [hits[0][0], hits[1][0]] == [1, 4]
    assert index.search("lactate septic shock", 1) == [hits[0]]  # the lower row survives the cut


def test_save_and_load_keep_scores(tmp_path, index):
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.n_docs == len(TEXTS) and loaded.vocab_size == index.vocab_size
    for query in ("lactate mortality", "ventilated patients", "kidney"):
        assert loaded.search(query, 10) == index.search(query, 10)
    assert BM25Index.load(tmp_path / "bm25", BM25Params()).n_docs == len(TEXTS)
    with pytest.raises(ValueError, match="other BM25 parameters"):
        BM25Index.load(tmp_path / "bm25", BM25Params(k1=1.2))


def test_without_stemming_and_empty_input():
    plain = BM25Index.build(TEXTS, BM25Params(stemmer=None))
    assert {row for row, _ in plain.search("ventilated", 10)} == {0}
    with pytest.raises(ValueError):
        BM25Index.build([])
