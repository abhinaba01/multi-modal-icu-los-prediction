"""Build the passage corpus from the downloaded sources (Stage 5).

    python -m fate_icu.rag.build_corpus --config configs/rag/smoke.yaml   # offline fixtures
    python -m fate_icu.rag.build_corpus --config configs/rag/full.yaml    # after fate_icu.rag.fetch

Reads the PubMed records (the efetch cache, restricted to ``pmids.json``), the StatPearls
archive and the guideline files, and chunks every document into passages of at most
``chunking.max_tokens`` MedCPT tokens. Writes ``<FATE_RAG_DIR>/<run_name>/corpus/<id>/``:
``passages.parquet``, ``corpus.json`` (content hash), ``corpus_manifest.csv`` (one row per
document: source, licence, PMID or URL, retrieval date), ``report.json`` and
``run_manifest.json``; then points ``<run_name>/corpus.CURRENT`` at it. The corpus id is the
first 12 characters of the content hash, so rebuilding identical inputs reuses the folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fate_icu.rag.chunking import HFTokenCounter, TokenCounter, WordCounter, chunk_documents
from fate_icu.rag.config import UNCONFIRMED, RagConfig, RagPaths, load_config, rag_paths, repo_path
from fate_icu.rag.corpus import Document, Passage, load_corpus_info, write_corpus
from fate_icu.rag.sources.guidelines import parse_guideline
from fate_icu.rag.sources.pubmed import exclusion_reason, iter_documents, parse_pubmed_xml, record_to_document
from fate_icu.rag.sources.statpearls import iter_statpearls_documents, parse_statpearls_article
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.env import Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest

MANIFEST_CSV = "corpus_manifest.csv"
MANIFEST_COLUMNS = [
    "source",
    "doc_id",
    "title",
    "year",
    "url",
    "pmid",
    "licence",
    "citation",
    "retrieved_on",
    "n_passages",
]


def load_counter(config: RagConfig) -> TokenCounter:
    """The token counter that sizes passages: the article encoder's tokenizer, or words."""
    if config.chunking.counter == "words":
        return WordCounter()
    from transformers import AutoTokenizer

    ref = config.models.article_encoder
    tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
    return HFTokenCounter(tokenizer, name=f"hf:{ref.repo}@{ref.revision[:8]}")


def archive_path(config: RagConfig, paths: RagPaths) -> Path:
    """Where ``fetch`` stores the StatPearls archive."""
    return paths.raw_statpearls / Path(urlsplit(config.sources.statpearls.url).path).name


def _pubmed(config: RagConfig, paths: RagPaths) -> tuple[list[Document], Counter[str], str]:
    source = config.sources.pubmed
    if source.fixtures:
        files = sorted(repo_path(source.fixtures).glob("*.xml"))
        if not files:
            raise FileNotFoundError(f"no PubMed fixture XML in {source.fixtures}")
        records = sorted((r for f in files for r in parse_pubmed_xml(f.read_bytes())), key=lambda r: r.pmid)
        pmids = [r.pmid for r in records]
        if len(pmids) != len(set(pmids)):
            raise ValueError("a PMID appears twice in the PubMed fixtures")
        counts: Counter[str] = Counter()
        documents = []
        for record in records:
            document = record_to_document(record, licence=source.licence)
            counts[exclusion_reason(record) or "kept"] += 1
            if document is not None:
                documents.append(document)
        return documents, counts, "fixture"
    collection_file = paths.raw_pubmed / "pmids.json"
    if not collection_file.exists():
        raise FileNotFoundError(f"{collection_file} is missing: run fate_icu.rag.fetch --source pubmed first")
    collection = json.loads(collection_file.read_text(encoding="utf-8"))
    documents, counts = iter_documents(paths.raw_pubmed, set(collection["pmids"]), licence=source.licence)
    return documents, counts, collection["retrieved_on"]


def _statpearls(config: RagConfig, paths: RagPaths) -> tuple[list[Document], Counter[str], str]:
    source = config.sources.statpearls
    if source.fixtures:
        files = sorted(repo_path(source.fixtures).glob("*.nxml"))
        if not files:
            raise FileNotFoundError(f"no StatPearls fixture .nxml in {source.fixtures}")
        counts: Counter[str] = Counter()
        documents = []
        for path in files[: source.max_articles]:
            document, reason = parse_statpearls_article(
                path.name,
                path.read_bytes(),
                licence_default=source.licence,
                exclude_titles=source.exclude_sections,
            )
            counts[reason or "kept"] += 1
            if document is not None:
                documents.append(document)
        return sorted(documents, key=lambda d: d.doc_id), counts, "fixture"
    archive = archive_path(config, paths)
    sidecar = archive.with_name(archive.name + ".json")
    if not archive.exists() or not sidecar.exists():
        raise FileNotFoundError(f"{archive} is missing: run fate_icu.rag.fetch --source statpearls first")
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    documents, counts = iter_statpearls_documents(
        archive,
        max_articles=source.max_articles,
        licence_default=source.licence,
        exclude_titles=source.exclude_sections,
    )
    return documents, counts, info["retrieved_on"]


def _guidelines(config: RagConfig, paths: RagPaths) -> tuple[list[Document], dict[str, str]]:
    source = config.sources.guidelines
    unconfirmed = [g.id for g in source.items if g.licence.strip().upper().startswith(UNCONFIRMED)]
    if unconfirmed:
        raise RuntimeError(
            "confirm the licence of " + ", ".join(unconfirmed) + " on the publisher's page and record it in "
            "the config (sources.guidelines.items[].licence) before building the corpus"
        )
    base = repo_path(source.dir) if source.dir else paths.raw_guidelines
    documents, retrieved = [], {}
    for spec in source.items:
        documents.append(parse_guideline(spec, base))
        path = Path(spec.file) if Path(spec.file).is_absolute() else base / spec.file
        retrieved[spec.id] = datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat()
    return documents, retrieved


def collect_documents(config: RagConfig, paths: RagPaths) -> tuple[list[Document], dict[str, Any], dict]:
    """Every enabled source's documents, the per-source report, and retrieval dates."""
    documents: list[Document] = []
    report: dict[str, Any] = {}
    retrieved: dict[tuple[str, str], str] = {}
    if config.sources.pubmed.enabled:
        docs, counts, when = _pubmed(config, paths)
        documents += docs
        report["pubmed"] = {"records": dict(sorted(counts.items())), "retrieved_on": when}
        retrieved.update({("pubmed", d.doc_id): d.meta.get("retrieved_on", when) for d in docs})
    if config.sources.statpearls.enabled:
        docs, counts, when = _statpearls(config, paths)
        documents += docs
        report["statpearls"] = {"articles": dict(sorted(counts.items())), "retrieved_on": when}
        retrieved.update({("statpearls", d.doc_id): when for d in docs})
    if config.sources.guidelines.enabled and config.sources.guidelines.items:
        docs, when_by_id = _guidelines(config, paths)
        documents += docs
        report["guidelines"] = {"documents": len(docs), "ids": [d.doc_id for d in docs]}
        retrieved.update({("guideline", d.doc_id): when_by_id[d.doc_id] for d in docs})
    if not documents:
        raise RuntimeError("no documents: every source is disabled or empty")
    return documents, report, retrieved


def write_manifest_csv(
    path: Path, documents: list[Document], passages: list[Passage], retrieved: dict[tuple[str, str], str]
) -> None:
    """``corpus_manifest.csv``: source, licence, PMID or URL and retrieval date per document."""
    n_passages = Counter((p.source, p.doc_id) for p in passages)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for d in sorted(documents, key=lambda d: (d.source, d.doc_id)):
            writer.writerow(
                {
                    "source": d.source,
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "year": d.year if d.year is not None else "",
                    "url": d.url,
                    "pmid": d.doc_id if d.source == "pubmed" else "",
                    "licence": d.licence,
                    "citation": d.citation,
                    "retrieved_on": retrieved.get((d.source, d.doc_id), ""),
                    "n_passages": n_passages[(d.source, d.doc_id)],
                }
            )


def run(config: RagConfig, *, settings: Settings | None = None, counter: TokenCounter | None = None) -> dict:
    """Build, write and report the corpus. ``counter`` can be injected (tests)."""
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    run_id = new_run_id(f"{config.run_name}_corpus")
    documents, source_report, retrieved = collect_documents(config, paths)
    counter = counter or load_counter(config)
    passages = chunk_documents(
        documents, counter, max_tokens=config.chunking.max_tokens, min_tokens=config.chunking.min_tokens
    )
    if not passages:
        raise RuntimeError("chunking produced no passages")

    staging = paths.build / "corpus" / f".staging-{run_id}"
    info = write_corpus(passages, staging)
    corpus_id = info.sha256[:12]
    final = paths.corpus(corpus_id)
    if final.exists() and load_corpus_info(final).sha256 == info.sha256:
        shutil.rmtree(staging)  # identical corpus already built: keep the existing folder
    else:
        if final.exists():
            raise RuntimeError(f"{final} exists with different content (hash prefix collision)")
        staging.replace(final)
    write_manifest_csv(final / MANIFEST_CSV, documents, passages, retrieved)
    without = len({(d.source, d.doc_id) for d in documents} - {(p.source, p.doc_id) for p in passages})
    report = {
        "run_id": run_id,
        "corpus_id": corpus_id,
        "corpus_sha256": info.sha256,
        "sources": source_report,
        "documents_by_source": info.documents_by_source,
        "passages_by_source": info.passages_by_source,
        "documents_without_passages": without,
        "tokens_per_passage": info.tokens,
        "chunking": {**config.chunking.model_dump(), "counter_name": counter.name},
    }
    (final / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_manifest(
        final,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        data={"corpus_sha256": info.sha256, "sources": source_report},
        model_revisions={"chunking_tokenizer": counter.name},
        extra={"machine": settings.machine},
    )
    paths.set_current("corpus", corpus_id)
    report["out_dir"] = str(final)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/smoke.yaml or full.yaml")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config))))


if __name__ == "__main__":
    main()
