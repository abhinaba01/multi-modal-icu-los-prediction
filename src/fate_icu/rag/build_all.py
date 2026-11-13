"""Every Stage 5 build step in one command, for a config that runs entirely on this machine.

    python -m fate_icu.rag.build_all --config configs/rag/smoke.yaml

corpus -> BM25 -> GPU step on the CPU (embeddings, FAISS, query bank) -> checks and
install -> example packets -> blind labelling sheet -> index check. The full corpus runs
its GPU step on Kaggle or Colab instead, so for configs/rag/full.yaml run the steps one by
one (docs/lab_runbook.md, Stage 5); the GPU step refuses a large corpus on a CPU.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from fate_icu.rag import build_corpus, build_index, gpu_step, verify
from fate_icu.rag.config import RagConfig, load_config, rag_paths
from fate_icu.rag.corpus import PASSAGES_FILE
from fate_icu.rag.embed import CPU_PASSAGE_LIMIT
from fate_icu.rag.evaluate import SHEET_FILE, make_sheet_from_bank
from fate_icu.rag.query_bank import BANK_FILE, BM25_HITS_FILE, load_bm25_rankings
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.device import get_device
from fate_icu.utils.env import Settings, load_settings


def run(config: RagConfig, *, settings: Settings | None = None, **injected: Any) -> dict[str, Any]:
    """All steps. ``injected`` may hold ``counter`` and the three encoders (tests)."""
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    report: dict[str, Any] = {
        "corpus": build_corpus.run(config, settings=settings, counter=injected.get("counter"))
    }
    report["bm25"] = build_index.run(config, "bm25", settings=settings)
    index_dir = paths.index(paths.current("index"))
    spec = json.loads((index_dir / gpu_step.GPU_SPEC_FILE).read_text(encoding="utf-8"))
    device = get_device(settings.device)
    if device == "cpu" and spec["n_passages"] > CPU_PASSAGE_LIMIT:
        raise SystemExit(
            f"{spec['n_passages']:,} passages are too many for a CPU: run the GPU step on Kaggle or Colab "
            "(docs/lab_runbook.md, Stage 5) and the remaining steps one by one"
        )
    report["gpu_step"] = gpu_step.run(
        spec,
        load_bm25_rankings(index_dir / BM25_HITS_FILE),
        paths.corpus(spec["corpus_id"]) / PASSAGES_FILE,
        index_dir / gpu_step.GPU_OUT_DIR,
        device=device,
        article_encoder=injected.get("article_encoder"),
        query_encoder=injected.get("query_encoder"),
        cross_encoder=injected.get("cross_encoder"),
    )
    report["dense"] = build_index.run(
        config,
        "dense",
        settings=settings,
        query_encoder=injected.get("query_encoder"),
        cross_encoder=injected.get("cross_encoder"),
    )
    report["labelling_sheet"] = make_sheet_from_bank(
        index_dir / BANK_FILE,
        paths.eval / SHEET_FILE,
        direction=config.evaluation.direction,
        depth=config.evaluation.pool_depth,
        seed=config.evaluation.seed,
    )
    report["verify"] = verify.run(config, settings=settings)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/smoke.yaml")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config))))


if __name__ == "__main__":
    main()
