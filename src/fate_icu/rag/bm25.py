"""BM25 over passages (bm25s) with a deterministic top k (Stage 5).

Each passage's ``search_text()`` (title, section title, text) is indexed. Tokens: lower
case, words of two or more characters, English stopwords removed, the Snowball English
stemmer (PyStemmer), so "ventilated" matches "ventilation". Scoring: the Lucene BM25
variant with k1 = 1.5 and b = 0.75. Only passages with a positive score count as
retrieved, and ties are broken by corpus row (= ``passage_id`` order).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import bm25s
import numpy as np
from pydantic import BaseModel, ConfigDict

INFO_FILE = "bm25_index.json"


class BM25Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    lower: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"
    stopwords: str = "english"
    stemmer: str | None = "english"  # a PyStemmer language, or None


def _stemmer(params: BM25Params):
    if params.stemmer is None:
        return None
    import Stemmer

    return Stemmer.Stemmer(params.stemmer).stemWords


def _tokenize(texts: Sequence[str], params: BM25Params, *, return_ids: bool):
    return bm25s.tokenize(
        list(texts),
        lower=params.lower,
        token_pattern=params.token_pattern,
        stopwords=params.stopwords,
        stemmer=_stemmer(params),
        return_ids=return_ids,
        show_progress=False,
    )


class BM25Index:
    """A bm25s index with exact, tie-stable search."""

    def __init__(self, retriever: bm25s.BM25, params: BM25Params, n_docs: int):
        self.retriever = retriever
        self.params = params
        self.n_docs = n_docs

    @classmethod
    def build(cls, texts: Sequence[str], params: BM25Params | None = None) -> BM25Index:
        params = params or BM25Params()
        if not texts:
            raise ValueError("no texts to index")
        retriever = bm25s.BM25(k1=params.k1, b=params.b, method=params.method)
        retriever.index(_tokenize(texts, params, return_ids=True), show_progress=False)
        return cls(retriever, params, len(texts))

    @property
    def vocab_size(self) -> int:
        return sum(1 for token in self.retriever.vocab_dict if token)

    def tokenize_query(self, text: str) -> list[str]:
        """The query tokens, through exactly the indexing pipeline."""
        return list(_tokenize([text], self.params, return_ids=False)[0])

    def scores(self, text: str) -> np.ndarray:
        """BM25 score of every passage (float32, length ``n_docs``)."""
        tokens = [t for t in self.tokenize_query(text) if t in self.retriever.vocab_dict]
        if not tokens:  # bm25s fails on an empty token list
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.retriever.get_scores(tokens), dtype=np.float32)

    def search(self, text: str, k: int) -> list[tuple[int, float]]:
        """The top ``k`` passages with a positive score, by (-score, row)."""
        scores = self.scores(text)
        positive = np.flatnonzero(scores > 0)
        order = np.lexsort((positive, -scores[positive]))[:k]
        return [(int(i), float(scores[i])) for i in positive[order]]

    def search_many(self, texts: Sequence[str], k: int) -> list[list[tuple[int, float]]]:
        return [self.search(text, k) for text in texts]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory), show_progress=False)
        info = {
            "params": self.params.model_dump(),
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "bm25s": metadata.version("bm25s"),
        }
        (directory / INFO_FILE).write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, params: BM25Params | None = None) -> BM25Index:
        """Load a saved index; if ``params`` is given it must equal the saved parameters."""
        directory = Path(directory)
        info = json.loads((directory / INFO_FILE).read_text(encoding="utf-8"))
        saved = BM25Params.model_validate(info["params"])
        if params is not None and params != saved:
            raise ValueError(f"the index in {directory} was built with other BM25 parameters: {saved}")
        retriever = bm25s.BM25.load(str(directory), show_progress=False)
        index = cls(retriever, saved, int(info["n_docs"]))
        if int(retriever.scores["num_docs"]) != index.n_docs:
            raise ValueError(
                f"{directory}: {retriever.scores['num_docs']} documents, expected {index.n_docs}"
            )
        return index
