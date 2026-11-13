"""Download the public Stage 5 sources (laptop, network).

    python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source statpearls --check-size
    python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source statpearls
    python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source pubmed

PubMed: every search in ``configs/rag/pubmed_queries.yaml`` is run through NCBI E-utilities
(date-sliced under the 10,000-ID cap), then the records are fetched in batches of 200 into
``<FATE_RAG_DIR>/raw/pubmed/``. It needs ``NCBI_EMAIL`` in ``.env`` (NCBI's usage policy);
``NCBI_API_KEY`` is optional. Both are sent only to NCBI and never printed or logged.
StatPearls: the NCBI Bookshelf archive into ``<FATE_RAG_DIR>/raw/statpearls/``.

Both steps resume where they stopped: run the same command again after an interruption.
Guideline PDFs are downloaded by hand (docs/lab_runbook.md, Stage 5).
"""

from __future__ import annotations

import argparse
from typing import Any

from fate_icu.rag.config import RagConfig, RagPaths, load_config, rag_paths, repo_path
from fate_icu.rag.sources.eutils import EutilsClient
from fate_icu.rag.sources.pubmed import collect_pmids, fetch_records, load_pubmed_queries
from fate_icu.rag.sources.statpearls import download, remote_size
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.env import Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest


def eutils_client(settings: Settings) -> EutilsClient:
    """An E-utilities client with the contact details from ``.env`` (fails loudly without them)."""
    if settings.ncbi_email is None:
        raise SystemExit(
            "NCBI_EMAIL is not set: add your contact email to .env (see .env.example). NCBI asks every "
            "E-utilities client for one; it is sent only to NCBI and never printed."
        )
    key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
    return EutilsClient(email=settings.ncbi_email.get_secret_value(), api_key=key)


def fetch_pubmed(config: RagConfig, paths: RagPaths, settings: Settings) -> dict[str, Any]:
    source = config.sources.pubmed
    if source.fixtures:
        return {"pubmed": f"uses the fixtures in {source.fixtures}; nothing to fetch"}
    client = eutils_client(settings)
    query_set = load_pubmed_queries(repo_path(source.queries))
    collection = collect_pmids(client, query_set, paths.raw_pubmed, max_records=source.max_records)
    report = fetch_records(
        client,
        collection.pmids,
        paths.raw_pubmed,
        batch_size=source.batch_size,
        max_workers=source.max_workers,
    )
    return {
        "pubmed": {
            "query_set_sha256": collection.query_set_sha256,
            "searches": len(collection.per_query),
            "ids_per_search": collection.per_query,
            "unique_pmids": len(collection.pmids),
            "retrieved_on": collection.retrieved_on,
            "fetch": report.model_dump() if hasattr(report, "model_dump") else vars(report),
            "api_key_used": settings.ncbi_api_key is not None,
        }
    }


def fetch_statpearls(config: RagConfig, paths: RagPaths, *, check_only: bool) -> dict[str, Any]:
    source = config.sources.statpearls
    if source.fixtures:
        return {"statpearls": f"uses the fixtures in {source.fixtures}; nothing to fetch"}
    size = remote_size(source.url)
    if source.expected_size is not None and size != source.expected_size:
        raise SystemExit(
            f"the StatPearls archive is now {size:,} bytes, not the {source.expected_size:,} in the config: "
            "NCBI has updated it. Review the change, then set sources.statpearls.expected_size."
        )
    if check_only:
        return {"statpearls": {"url": source.url, "remote_size_bytes": size}}
    info = download(source.url, paths.raw_statpearls, expected_size=source.expected_size)
    return {"statpearls": info.model_dump(mode="json") if hasattr(info, "model_dump") else vars(info)}


def run(
    config: RagConfig, source: str, *, check_only: bool = False, settings: Settings | None = None
) -> dict:
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    run_id = new_run_id(f"{config.run_name}_fetch_{source}")
    report: dict[str, Any] = {"run_id": run_id}
    if source in ("statpearls", "all"):
        report.update(fetch_statpearls(config, paths, check_only=check_only))
    if source in ("pubmed", "all") and not check_only:
        report.update(fetch_pubmed(config, paths, settings))
    if not check_only:
        write_run_manifest(
            paths.runs / run_id,
            run_id=run_id,
            config=config.model_dump(),
            seeds=[config.seed],
            data=report,
            extra={"machine": settings.machine},
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/full.yaml")
    parser.add_argument("--source", required=True, choices=["pubmed", "statpearls", "all"])
    parser.add_argument("--check-size", action="store_true", help="StatPearls: report the remote size only")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config), args.source, check_only=args.check_size)))


if __name__ == "__main__":
    main()
