"""Check a copied Stage 5 index against its recorded hashes (run in the lab after copying).

    python -m fate_icu.rag.verify --config configs/rag/full.yaml

Recomputes the corpus content hash and the SHA-256 of every file listed in ``index.json``
(BM25, embeddings, FAISS, query bank) and checks that the query bank belongs to this index
and corpus. Exits with an error on any mismatch or missing file.
"""

from __future__ import annotations

import argparse
from typing import Any

from fate_icu.rag.build_index import read_index_info
from fate_icu.rag.config import RagConfig, load_config, rag_paths
from fate_icu.rag.corpus import verify_corpus
from fate_icu.rag.query_bank import BANK_FILE
from fate_icu.rag.retrieve import load_query_bank
from fate_icu.rag.version import binary_sha256
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.env import Settings, load_settings


def run(config: RagConfig, *, version: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    version = version or paths.current("index")
    index_dir = paths.index(version)
    info = read_index_info(index_dir)
    corpus = verify_corpus(paths.corpus(info["corpus_id"]))
    problems = []
    if corpus.sha256 != info["corpus_sha256"]:
        problems.append("the corpus hash differs from index.json")
    for name, expected in sorted(info["files"].items()):
        path = index_dir / name
        if not path.exists():
            problems.append(f"missing {name}")
        elif binary_sha256(path) != expected:
            problems.append(f"changed {name}")
    bank_path = index_dir / BANK_FILE
    if bank_path.exists():
        bank = load_query_bank(bank_path)
        if bank.index_version != version or bank.corpus_sha256 != info["corpus_sha256"]:
            problems.append("the query bank belongs to another index or corpus")
        if info.get("query_bank_sha256") not in (None, bank.sha256):
            problems.append("the query bank content differs from index.json")
    if problems:
        raise SystemExit("index check failed: " + "; ".join(problems))
    return {
        "index_version": version,
        "corpus_sha256": corpus.sha256,
        "passages": corpus.n_passages,
        "files_checked": len(info["files"]),
        "query_bank": bank_path.exists(),
        "ok": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/full.yaml")
    parser.add_argument("--version", help="index version (default: the current one)")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config), version=args.version)))


if __name__ == "__main__":
    main()
