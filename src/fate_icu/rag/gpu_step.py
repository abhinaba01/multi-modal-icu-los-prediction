"""The GPU step: passage embeddings, the FAISS index and the query bank (Stage 5).

Full corpus, on a Kaggle or Colab GPU, from the files that ``gpu_bundle`` collected
(``gpu_spec.json``, ``bm25_hits.json``, ``passages.parquet``):

    python -m fate_icu.rag.gpu_step --bundle /kaggle/input/<dataset> --out /kaggle/working/gpu_out

Smoke corpus, on the laptop CPU, with every path taken from the config:

    python -m fate_icu.rag.gpu_step --config configs/rag/smoke.yaml

It embeds the passages (resumable shards), builds the exact FAISS index, encodes the 100
queries, fuses dense and BM25 lists, reranks them with the cross-encoder and writes
``<out>/dense/`` (embeddings, ``faiss.index``), ``<out>/query_bank.json`` and
``<out>/gpu_run.json``. Then ``build_index --part dense`` checks and installs the output on
the laptop. The corpus is re-hashed first and must match ``gpu_spec.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from fate_icu.rag.corpus import PASSAGES_FILE, load_corpus
from fate_icu.rag.dense import EMBEDDINGS_INFO_FILE, SHARDS_DIR, DenseIndex, load_embeddings
from fate_icu.rag.embed import CPU_PASSAGE_LIMIT, DENSE_DIR, embed, embedding_model_info
from fate_icu.rag.encoders import CrossEncoder, ModelRef, QueryEncoder
from fate_icu.rag.query_bank import BANK_FILE, BM25_HITS_FILE, build_bank, load_bm25_rankings
from fate_icu.rag.retrieve import save_query_bank
from fate_icu.rag.version import binary_sha256
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.device import get_device

GPU_SPEC_FILE = "gpu_spec.json"
GPU_OUT_DIR = "gpu_out"
FAISS_FILE = "faiss.index"
RUN_FILE = "gpu_run.json"


def run(
    spec: dict[str, Any],
    bm25_hits: dict[str, Any],
    corpus_path: Path,
    out_dir: Path,
    *,
    device: str,
    article_encoder: Any = None,
    query_encoder: Any = None,
    cross_encoder: Any = None,
    keep_shards: bool = False,
    log=print,
) -> dict[str, Any]:
    """Embed, index and build the bank into ``out_dir``. Encoders can be injected (tests)."""
    started = time.monotonic()
    out_dir = Path(out_dir)
    passages = load_corpus(corpus_path)
    embedded = embed(passages, spec, out_dir / DENSE_DIR, device=device, encoder=article_encoder, log=log)
    embeddings = load_embeddings(
        out_dir / DENSE_DIR,
        expect_corpus_sha256=spec["corpus_sha256"],
        expect_model=embedding_model_info(spec["embedding"]),
    )
    dense = DenseIndex.build(embeddings)
    dense.save(out_dir / DENSE_DIR / FAISS_FILE)
    if not keep_shards:
        shutil.rmtree(out_dir / DENSE_DIR / SHARDS_DIR, ignore_errors=True)  # they duplicate embeddings.npy
    bank_spec = spec["bank"]
    query_encoder = query_encoder or QueryEncoder.from_ref(
        ModelRef(**bank_spec["query_encoder"]), device=device, max_length=bank_spec["query_max_length"]
    )
    cross_encoder = cross_encoder or CrossEncoder.from_ref(
        ModelRef(**bank_spec["cross_encoder"]),
        device=device,
        max_length=bank_spec["cross_max_length"],
        batch_size=bank_spec["cross_batch_size"],
    )
    log(f"building the query bank: {len(bank_spec['queries'])} queries")
    bank = build_bank(
        passages,
        bank_spec["queries"],
        bm25_hits,
        dense,
        query_encoder,
        cross_encoder,
        params={k: bank_spec[k] for k in ("depth", "rrf_k", "rerank_depth", "store_depth")}
        | {"models": {k: bank_spec[k] for k in ("query_encoder", "article_encoder", "cross_encoder")}},
        index_version=spec["index_version"],
        corpus_sha256=spec["corpus_sha256"],
        query_spec_sha256=bank_spec["query_spec_sha256"],
        template=bank_spec["template"],
    )
    save_query_bank(bank, out_dir / BANK_FILE)
    summary = {
        "index_version": spec["index_version"],
        "device": device,
        "passages": len(passages),
        "embedding_seconds": embedded["seconds"],
        "total_seconds": round(time.monotonic() - started, 1),
        "embeddings_sha256": json.loads((out_dir / DENSE_DIR / EMBEDDINGS_INFO_FILE).read_text())["sha256"],
        "faiss_sha256": binary_sha256(out_dir / DENSE_DIR / FAISS_FILE),
        "query_bank_sha256": bank.sha256,
        "packages": {
            name: metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "faiss-cpu", "numpy")
        },
    }
    (out_dir / RUN_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="laptop: configs/rag/smoke.yaml (paths from the config)")
    parser.add_argument(
        "--bundle", help="GPU machine: the folder with gpu_spec.json, bm25_hits.json, passages.parquet"
    )
    parser.add_argument("--out", help="GPU machine: output folder")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-long-cpu-run", action="store_true", help="embed a large corpus on a CPU anyway"
    )
    parser.add_argument(
        "--keep-shards", action="store_true", help="keep the embedding shards (they duplicate the result)"
    )
    args = parser.parse_args(argv)
    device = get_device(args.device)
    if args.config:
        from fate_icu.rag.config import load_config, rag_paths
        from fate_icu.utils.env import load_settings

        config = load_config(args.config)
        paths = rag_paths(config, load_settings())
        index_dir = paths.index(paths.current("index"))
        spec = json.loads((index_dir / GPU_SPEC_FILE).read_text(encoding="utf-8"))
        bm25_path, corpus_path = index_dir / BM25_HITS_FILE, paths.corpus(spec["corpus_id"]) / PASSAGES_FILE
        out_dir = index_dir / GPU_OUT_DIR
    elif args.bundle and args.out:
        bundle = Path(args.bundle)
        spec = json.loads((bundle / GPU_SPEC_FILE).read_text(encoding="utf-8"))
        bm25_path, corpus_path, out_dir = bundle / BM25_HITS_FILE, bundle / PASSAGES_FILE, Path(args.out)
    else:
        parser.error("give --config, or both --bundle and --out")
    if device == "cpu" and spec["n_passages"] > CPU_PASSAGE_LIMIT and not args.allow_long_cpu_run:
        raise SystemExit(
            f"{spec['n_passages']:,} passages would take hours on a CPU: "
            "run this step on a Kaggle or Colab GPU "
            "(docs/lab_runbook.md, Stage 5), or pass --allow-long-cpu-run"
        )
    summary = run(
        spec, load_bm25_rankings(bm25_path), corpus_path, out_dir, device=device, keep_shards=args.keep_shards
    )
    print(format_aggregates(summary))


if __name__ == "__main__":
    main()
