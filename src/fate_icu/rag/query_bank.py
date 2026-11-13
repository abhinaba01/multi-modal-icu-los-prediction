"""Build the query bank, and check a bank built on another machine (Stage 5).

For each of the 100 queries (50 driver keys x longer/shorter) the bank stores four rankings:

* ``bm25``: BM25 top ``depth`` (computed on the laptop, shipped to the GPU as ``bm25_hits.json``);
* ``dense``: the MedCPT query encoder against the FAISS index, top ``depth``;
* ``hybrid``: reciprocal-rank fusion of those two lists (k = ``rrf_k``);
* ``hybrid_rerank``: the first ``rerank_depth`` fused candidates rescored by the MedCPT
  cross-encoder on ``"Title. Text"``.

``retrieve(packet)`` only looks queries up in the bank, so retrieval is identical on every
machine and never needs a model at explanation time. Ties are broken by corpus row.

A bank built on the GPU is checked on the laptop before it is installed: BM25 lists must be
identical, the dense lists are recomputed for every query on the CPU, and the reranking is
recomputed for a few queries; small float differences between machines may swap near-ties.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fate_icu.rag.corpus import Passage
from fate_icu.rag.dense import DenseIndex
from fate_icu.rag.fusion import rerank, rrf
from fate_icu.rag.retrieve import SYSTEMS, BankQuery, QueryBank, RankedHit, Retriever

BANK_FILE = "query_bank.json"
BM25_HITS_FILE = "bm25_hits.json"
MIN_DENSE_AGREEMENT = 0.9  # mean overlap of the stored and recomputed dense top lists
NEAR_TIE = 1e-3  # cross-encoder logits closer than this may swap between machines


def _hits(ranked: Sequence[tuple[int, float]], passages: Sequence[Passage]) -> list[RankedHit]:
    return [RankedHit(passage_id=passages[i].passage_id, score=float(s)) for i, s in ranked]


def build_bank(
    passages: Sequence[Passage],
    queries: Sequence[Mapping[str, Any]],
    bm25_hits: Mapping[str, Sequence[Sequence[float]]],
    dense: DenseIndex,
    query_encoder: Any,
    cross_encoder: Any,
    *,
    params: Mapping[str, Any],
    index_version: str,
    corpus_sha256: str,
    query_spec_sha256: str,
    template: str,
) -> QueryBank:
    """The query bank (every component injected, so the same code runs on the CPU and the GPU).

    Args:
        queries: QueryDef dumps (query_id, modality, key, direction, text).
        bm25_hits: ``{query_id: [[row, score], ...]}`` from :func:`bm25_rankings`.
        params: depth, rrf_k, rerank_depth, store_depth and the model references.
    """
    if len(passages) != dense.ntotal:
        raise RuntimeError(f"{len(passages)} passages but {dense.ntotal} vectors")
    depth, rerank_depth, store_depth = (
        int(params["depth"]),
        int(params["rerank_depth"]),
        int(params["store_depth"]),
    )
    vectors = query_encoder.encode([q["text"] for q in queries])
    dense_hits = dense.search(vectors, depth)
    bank_queries: dict[str, BankQuery] = {}
    keep: set[int] = set()
    for query, dense_ranked in zip(queries, dense_hits, strict=True):
        bm25_ranked = [(int(row), float(score)) for row, score in bm25_hits[query["query_id"]]][:depth]
        fused = rrf([[i for i, _ in bm25_ranked], [i for i, _ in dense_ranked]], k=int(params["rrf_k"]))
        candidates = [i for i, _ in fused[:rerank_depth]]
        reranked = rerank(
            query["text"],
            candidates,
            [passages[i].cross_text() for i in candidates],
            cross_encoder,
            depth=rerank_depth,
        )
        ranked = {"bm25": bm25_ranked, "dense": dense_ranked, "hybrid": fused, "hybrid_rerank": reranked}
        for hits in ranked.values():
            keep.update(i for i, _ in hits[:store_depth])
        bank_queries[query["query_id"]] = BankQuery(
            query_id=query["query_id"],
            modality=query["modality"],
            key=query["key"],
            direction=query["direction"],
            text=query["text"],
            rankings={system: _hits(ranked[system], passages) for system in SYSTEMS},
        )
    return QueryBank(
        index_version=index_version,
        corpus_sha256=corpus_sha256,
        query_spec_sha256=query_spec_sha256,
        template=template,
        params=dict(params),
        queries=bank_queries,
        passages={passages[i].passage_id: passages[i] for i in sorted(keep)},
    )


def bm25_rankings(
    bm25: Any, queries: Sequence[Mapping[str, Any]], depth: int
) -> dict[str, list[list[float]]]:
    """``{query_id: [[row, score], ...]}``: the BM25 lists shipped to the GPU step."""
    return {q["query_id"]: [[row, score] for row, score in bm25.search(q["text"], depth)] for q in queries}


def save_bm25_rankings(path: Path, rankings: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(rankings, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def load_bm25_rankings(path: Path) -> dict[str, list[list[float]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_bank(
    bank: QueryBank,
    passages: Sequence[Passage],
    bm25_hits: Mapping[str, Sequence[Sequence[float]]],
    dense: DenseIndex,
    query_encoder: Any,
    cross_encoder: Any,
    *,
    rerank_queries: int = 2,
) -> dict[str, Any]:
    """Recompute parts of a bank on this machine and compare (fails loudly on disagreement).

    Raises:
        RuntimeError: BM25 lists differ, the dense lists agree on average less than
            ``MIN_DENSE_AGREEMENT``, or a rechecked rerank order differs beyond near-ties.
    """
    params = bank.params
    depth, store_depth = int(params["depth"]), int(params["store_depth"])
    row_of = {p.passage_id: i for i, p in enumerate(passages)}
    queries = list(bank.queries.values())
    for query in queries:
        stored = [(row_of[h.passage_id], h.score) for h in query.rankings["bm25"]]
        expected = [(int(r), float(s)) for r, s in bm25_hits[query.query_id]][:depth]
        if [r for r, _ in stored] != [r for r, _ in expected]:
            raise RuntimeError(f"{query.query_id}: the bank's BM25 list differs from this machine's")
    recomputed = dense.search(query_encoder.encode([q.text for q in queries]), depth)
    overlaps = []
    for query, hits in zip(queries, recomputed, strict=True):
        stored = {h.passage_id for h in query.rankings["dense"][:store_depth]}
        again = {passages[i].passage_id for i, _ in hits[:store_depth]}
        overlaps.append(len(stored & again) / max(1, len(stored | again)))
    agreement = statistics.mean(overlaps) if overlaps else 1.0
    if agreement < MIN_DENSE_AGREEMENT:
        raise RuntimeError(f"dense rankings agree only {agreement:.2f} with this machine's (another model?)")
    checked = []
    for query in queries[:rerank_queries]:
        candidates = [row_of[h.passage_id] for h in query.rankings["hybrid"][: int(params["rerank_depth"])]]
        scores = cross_encoder.score(query.text, [passages[i].cross_text() for i in candidates])
        stored = query.rankings["hybrid_rerank"]
        mine = dict(zip((passages[i].passage_id for i in candidates), scores, strict=True))
        for hit in stored:
            if abs(mine[hit.passage_id] - hit.score) > max(NEAR_TIE, 1e-3 * abs(hit.score)):
                raise RuntimeError(
                    f"{query.query_id}: cross-encoder score of {hit.passage_id} differs on this machine"
                )
        checked.append(query.query_id)
    return {
        "bm25_identical": True,
        "dense_mean_overlap_at_store_depth": round(agreement, 4),
        "dense_queries_identical": sum(o == 1.0 for o in overlaps),
        "rerank_rechecked": checked,
    }


def bank_report(bank: QueryBank, top_k: int) -> dict[str, Any]:
    """Aggregates over the bank: hits per system, sources of the passages a driver receives."""
    queries = list(bank.queries.values())
    per_system = {
        system: {
            "queries_with_fewer_than_top_k": sum(len(q.rankings[system]) < top_k for q in queries),
            "median_hits": statistics.median(len(q.rankings[system]) for q in queries),
        }
        for system in SYSTEMS
    }
    sources = Counter(
        bank.passages[h.passage_id].source for q in queries for h in q.rankings["hybrid_rerank"][:top_k]
    )
    overlap = [
        len(
            {h.passage_id for h in q.rankings["bm25"][:10]} & {h.passage_id for h in q.rankings["dense"][:10]}
        )
        for q in queries
    ]
    return {
        "queries": len(queries),
        "stored_passages": len(bank.passages),
        "per_system": per_system,
        "sources_of_top_k_reranked": dict(sorted(sources.items())),
        "median_bm25_dense_overlap_at_10": statistics.median(overlap) if overlap else 0,
    }


def retrieve_examples(bank: QueryBank, packets_dir: Path, top_k: int) -> dict[str, Any]:
    """Run ``retrieve`` on the example packets twice and check the results are identical."""
    retriever = Retriever(bank, top_k=top_k)
    out = {}
    for path in sorted(Path(packets_dir).glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        first, second = retriever.retrieve(packet), retriever.retrieve(packet)
        if first.model_dump_json() != second.model_dump_json():
            raise RuntimeError(f"retrieve() is not deterministic on {path.name}")
        out[path.name] = {
            "drivers": len(first.by_driver),
            "passages": len(first.passages),
            "rids_by_driver": {d: list(r) for d, r in first.by_driver.items()},
        }
    return out
