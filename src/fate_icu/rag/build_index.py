"""Build the BM25 index, then check and install the GPU step's output (Stage 5, laptop).

    python -m fate_icu.rag.build_index --config configs/rag/full.yaml --part bm25
    # GPU step: python -m fate_icu.rag.gpu_step (Kaggle or Colab for the full corpus)
    python -m fate_icu.rag.build_index --config configs/rag/full.yaml --part dense

``--part bm25`` builds BM25 over the current corpus and writes, in
``<FATE_RAG_DIR>/<run_name>/index/<version>/``: ``bm25/``, ``bm25_hits.json`` (BM25 top
lists of all queries), ``gpu_spec.json`` (everything the GPU step needs) and ``index.json``.

``--part dense`` takes the GPU output from ``<index>/gpu_out/``: it checks the embeddings
(corpus hash, model, SHA-256), rebuilds the FAISS index here, checks the query bank
against this machine (identical BM25 lists, recomputed dense lists, a few recomputed
reranks), and installs ``dense/`` and ``query_bank.json``. ``index.json`` then records the
SHA-256 of every file, which ``fate_icu.rag.verify`` re-checks in the lab.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from fate_icu.rag.bm25 import BM25Index
from fate_icu.rag.config import RagConfig, RagPaths, load_config, rag_paths, repo_path
from fate_icu.rag.corpus import load_corpus, verify_corpus
from fate_icu.rag.dense import EMBEDDINGS_FILE, EMBEDDINGS_INFO_FILE, DenseIndex, load_embeddings
from fate_icu.rag.embed import DENSE_DIR, embedding_model_info
from fate_icu.rag.encoders import CrossEncoder, QueryEncoder
from fate_icu.rag.gpu_step import FAISS_FILE, GPU_OUT_DIR, GPU_SPEC_FILE, RUN_FILE
from fate_icu.rag.queries import build_queries, load_query_spec
from fate_icu.rag.query_bank import (
    BANK_FILE,
    BM25_HITS_FILE,
    bank_report,
    bm25_rankings,
    check_bank,
    load_bm25_rankings,
    retrieve_examples,
    save_bm25_rankings,
)
from fate_icu.rag.retrieve import load_query_bank, save_query_bank
from fate_icu.rag.version import INDEX_INFO_FILE, binary_sha256, files_sha256, index_payload, index_version
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.device import get_device
from fate_icu.utils.env import Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest

BM25_DIR = "bm25"


def read_index_info(index_dir: Path) -> dict[str, Any]:
    path = index_dir / INDEX_INFO_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run build_index --part bm25 first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_index_info(index_dir: Path, info: dict[str, Any]) -> None:
    tmp = index_dir / (INDEX_INFO_FILE + ".tmp")
    tmp.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, index_dir / INDEX_INFO_FILE)


def gpu_spec(version: str, corpus_id: str, corpus: Any, config: RagConfig, spec: Any) -> dict[str, Any]:
    """Everything the GPU step needs, so it runs without the repository's configs."""
    r, m = config.retrieval, config.models
    return {
        "index_version": version,
        "corpus_id": corpus_id,
        "corpus_sha256": corpus.sha256,
        "n_passages": corpus.n_passages,
        "embedding": {
            "model": m.article_encoder.model_dump(),
            "max_length": config.embedding.max_length,
            "batch_size": config.embedding.batch_size,
            "shard_size": config.embedding.shard_size,
            "autocast_fp16": config.embedding.autocast_fp16,
        },
        "bank": {
            "query_encoder": m.query_encoder.model_dump(),
            "article_encoder": m.article_encoder.model_dump(),
            "cross_encoder": m.cross_encoder.model_dump(),
            "depth": r.depth,
            "rrf_k": r.rrf_k,
            "rerank_depth": r.rerank_depth,
            "store_depth": r.store_depth,
            "query_max_length": r.query_max_length,
            "cross_max_length": r.cross_max_length,
            "cross_batch_size": r.cross_batch_size,
            "query_spec_sha256": spec.sha256,
            "template": spec.template,
            "queries": [q.model_dump() for q in build_queries(spec)],
        },
    }


def build_bm25(config: RagConfig, paths: RagPaths, settings: Settings, corpus_id: str | None = None) -> dict:
    corpus_id = corpus_id or paths.current("corpus")
    corpus_dir = paths.corpus(corpus_id)
    corpus = verify_corpus(corpus_dir)  # recomputes the content hash
    passages = load_corpus(corpus_dir)
    version = index_version(corpus.sha256, config)
    out = paths.index(version)
    out.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Index.build([p.search_text() for p in passages], config.bm25)
    bm25.save(out / BM25_DIR)
    query_spec = load_query_spec(repo_path(config.queries))
    spec = gpu_spec(version, corpus_id, corpus, config, query_spec)
    save_bm25_rankings(
        out / BM25_HITS_FILE, bm25_rankings(bm25, spec["bank"]["queries"], config.retrieval.depth)
    )
    (out / GPU_SPEC_FILE).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    info = {
        "version": version,
        "corpus_id": corpus_id,
        "corpus_sha256": corpus.sha256,
        "n_passages": corpus.n_passages,
        "query_spec_sha256": query_spec.sha256,
        "covers": index_payload(corpus.sha256, config),
        "files": files_sha256(
            out, [*sorted((out / BM25_DIR).iterdir()), out / BM25_HITS_FILE, out / GPU_SPEC_FILE]
        ),
    }
    write_index_info(out, info)
    run_id = new_run_id(f"{config.run_name}_bm25")
    write_run_manifest(
        out / "runs" / run_id,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        data={"corpus_sha256": corpus.sha256, "index_version": version},
        extra={"machine": settings.machine, "part": "bm25"},
    )
    paths.set_current("index", version)
    no_hits = sum(1 for hits in load_bm25_rankings(out / BM25_HITS_FILE).values() if not hits)
    return {
        "index_version": version,
        "corpus_id": corpus_id,
        "passages": corpus.n_passages,
        "bm25_vocabulary": bm25.vocab_size,
        "queries_without_bm25_hits": no_hits,
        "next_step": "the GPU step (fate_icu.rag.gpu_step), then build_index --part dense",
        "out_dir": str(out),
    }


def install_dense(
    config: RagConfig,
    paths: RagPaths,
    settings: Settings,
    *,
    gpu_out: Path | None = None,
    query_encoder: Any = None,
    cross_encoder: Any = None,
) -> dict:
    """Check the GPU output against this machine and install it (encoders injectable for tests)."""
    version = paths.current("index")
    out = paths.index(version)
    info = read_index_info(out)
    spec = json.loads((out / GPU_SPEC_FILE).read_text(encoding="utf-8"))
    gpu_out = gpu_out or out / GPU_OUT_DIR
    if not (gpu_out / BANK_FILE).exists():
        raise FileNotFoundError(
            f"{gpu_out / BANK_FILE} is missing: run the GPU step and copy its output there"
        )
    embeddings = load_embeddings(
        gpu_out / DENSE_DIR,
        expect_corpus_sha256=info["corpus_sha256"],
        expect_model=embedding_model_info(spec["embedding"]),
    )
    if embeddings.shape[0] != info["n_passages"]:
        raise RuntimeError(f"{embeddings.shape[0]} embeddings for {info['n_passages']} passages")
    dense = DenseIndex.build(embeddings)  # rebuilt here rather than trusting the copied index
    bank = load_query_bank(gpu_out / BANK_FILE)
    if (bank.index_version, bank.corpus_sha256) != (version, info["corpus_sha256"]):
        raise RuntimeError("the query bank was built for another index or corpus")
    query_spec = load_query_spec(repo_path(config.queries))
    if bank.query_spec_sha256 != query_spec.sha256:
        raise RuntimeError(
            "configs/rag/queries.yaml changed since the GPU step: rebuild with build_index --part bm25"
        )
    passages = load_corpus(paths.corpus(info["corpus_id"]))
    device = get_device(settings.device)
    b = spec["bank"]
    query_encoder = query_encoder or QueryEncoder.from_ref(
        config.models.query_encoder, device=device, max_length=b["query_max_length"]
    )
    cross_encoder = cross_encoder or CrossEncoder.from_ref(
        config.models.cross_encoder,
        device=device,
        max_length=b["cross_max_length"],
        batch_size=b["cross_batch_size"],
    )
    checks = check_bank(
        bank,
        passages,
        load_bm25_rankings(out / BM25_HITS_FILE),
        dense,
        query_encoder,
        cross_encoder,
        rerank_queries=config.retrieval.cpu_check_queries,
    )
    dense_dir = out / DENSE_DIR
    dense_dir.mkdir(exist_ok=True)
    for name in (EMBEDDINGS_FILE, EMBEDDINGS_INFO_FILE):
        shutil.copy2(gpu_out / DENSE_DIR / name, dense_dir / name)
    dense.save(dense_dir / FAISS_FILE)
    save_query_bank(bank, out / BANK_FILE)
    if (gpu_out / RUN_FILE).exists():
        shutil.copy2(gpu_out / RUN_FILE, out / RUN_FILE)
    info["files"].update(
        files_sha256(
            out, [dense_dir / EMBEDDINGS_FILE, dense_dir / EMBEDDINGS_INFO_FILE, dense_dir / FAISS_FILE]
        )
    )
    info["files"][BANK_FILE] = binary_sha256(out / BANK_FILE)
    info["query_bank_sha256"] = bank.sha256
    info["embedding_dim"] = int(embeddings.shape[1])
    write_index_info(out, info)
    report = {
        "index_version": version,
        "vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "query_bank_sha256": bank.sha256,
        "checks_against_this_machine": checks,
        **bank_report(bank, config.retrieval.top_k),
    }
    if config.smoke_packets:
        report["example_packets"] = retrieve_examples(
            bank, repo_path(config.smoke_packets), config.retrieval.top_k
        )
    run_id = new_run_id(f"{config.run_name}_dense")
    write_run_manifest(
        out / "runs" / run_id,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        data={
            "corpus_sha256": info["corpus_sha256"],
            "index_version": version,
            "query_bank_sha256": bank.sha256,
        },
        model_revisions={
            name: f"{ref['repo']}@{ref['revision']}" for name, ref in bank.params["models"].items()
        },
        device=device,
        extra={"machine": settings.machine, "part": "dense", "checks": checks},
    )
    report["out_dir"] = str(out)
    return report


def run(config: RagConfig, part: str, *, settings: Settings | None = None, **kw: Any) -> dict:
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    if part == "bm25":
        return build_bm25(config, paths, settings)
    if part == "dense":
        return install_dense(config, paths, settings, **kw)
    raise ValueError(f"unknown part {part!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/smoke.yaml or full.yaml")
    parser.add_argument("--part", required=True, choices=["bm25", "dense"])
    parser.add_argument("--gpu-out", help="the GPU step's output folder (default: <index>/gpu_out)")
    args = parser.parse_args(argv)
    kw = {"gpu_out": Path(args.gpu_out)} if args.gpu_out else {}
    print(format_aggregates(run(load_config(args.config), args.part, **kw)))


if __name__ == "__main__":
    main()
