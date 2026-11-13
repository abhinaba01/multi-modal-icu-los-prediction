"""Stage 5 end to end, offline: fixtures -> corpus -> BM25 -> GPU step -> checks -> retrieve -> verify.

Tiny random models stand in for MedCPT (the smoke run uses the real ones), and a word
counter for the MedCPT tokenizer, so this runs in seconds without downloads.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from rag_tiny_models import tiny_cross_encoder, tiny_encoder, tiny_tokenizer

from fate_icu.rag import build_all, build_corpus, build_index, fetch, gpu_bundle, verify
from fate_icu.rag.bm25 import BM25Params
from fate_icu.rag.chunking import WordCounter
from fate_icu.rag.config import RagConfig, load_config, rag_paths
from fate_icu.rag.encoders import ArticleEncoder, CrossEncoder, ModelRef, QueryEncoder
from fate_icu.rag.gpu_step import GPU_OUT_DIR
from fate_icu.rag.query_bank import BANK_FILE, BM25_HITS_FILE
from fate_icu.rag.retrieve import Retriever, load_query_bank, save_query_bank
from fate_icu.rag.version import index_version
from fate_icu.utils.env import REPO_ROOT, load_settings


def _settings(tmp_path, monkeypatch, name="rag"):
    monkeypatch.setenv("FATE_RAG_DIR", str(tmp_path / name))
    monkeypatch.setenv("FATE_DEVICE", "cpu")
    for key in ("NCBI_EMAIL", "NCBI_API_KEY", "OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(key, raising=False)
    return load_settings(env_file=tmp_path / "none.env")


def _tiny():
    tokenizer = tiny_tokenizer()
    return {
        "counter": WordCounter(),
        "article_encoder": ArticleEncoder(tiny_encoder(0), tokenizer),
        "query_encoder": QueryEncoder(tiny_encoder(0), tokenizer),
        "cross_encoder": CrossEncoder(tiny_cross_encoder(1), tokenizer),
    }


@pytest.fixture
def built(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    config = load_config("configs/rag/smoke.yaml")
    return config, settings, build_all.run(config, settings=settings, **_tiny())


def test_smoke_pipeline_end_to_end(built):
    config, settings, report = built
    corpus = report["corpus"]
    assert corpus["documents_by_source"] == {"guideline": 1, "pubmed": 6, "statpearls": 3}
    assert set(corpus["passages_by_source"]) == {"guideline", "pubmed", "statpearls"}
    assert (
        corpus["sources"]["pubmed"]["records"]["kept"] == 6 and corpus["chunking"]["counter_name"] == "words"
    )
    checks = report["dense"]["checks_against_this_machine"]
    assert checks["bm25_identical"] and checks["dense_mean_overlap_at_store_depth"] == 1.0
    assert len(checks["rerank_rechecked"]) == 2
    examples = report["dense"]["example_packets"]
    assert examples["packet_long.json"]["drivers"] == 3
    short = examples["packet_short.json"]["rids_by_driver"]
    assert short["D1"] == short["D3"]  # two lactate drivers share their passages
    assert report["labelling_sheet"]["queries"] == 50 and report["verify"]["ok"]
    paths = rag_paths(config, settings)
    corpus_dir = paths.corpus(paths.current("corpus"))
    assert (corpus_dir / "corpus_manifest.csv").read_text(encoding="utf-8").startswith("source,doc_id,title")
    bank = load_query_bank(paths.index(paths.current("index")) / BANK_FILE)
    assert len(bank.queries) == 100 and bank.index_version == paths.current("index")


def test_rebuilding_gives_identical_hashes(built, tmp_path, monkeypatch):
    config, _, first = built
    settings = _settings(tmp_path, monkeypatch, name="rag_again")
    second = build_all.run(config, settings=settings, **_tiny())
    assert first["corpus"]["corpus_sha256"] == second["corpus"]["corpus_sha256"]
    assert first["bm25"]["index_version"] == second["bm25"]["index_version"]
    assert first["dense"]["query_bank_sha256"] == second["dense"]["query_bank_sha256"]


def test_verify_and_install_catch_tampering(built):
    config, settings, _ = built
    paths = rag_paths(config, settings)
    index_dir = paths.index(paths.current("index"))
    bank = load_query_bank(index_dir / GPU_OUT_DIR / BANK_FILE)
    save_query_bank(bank.model_copy(update={"index_version": "another"}), index_dir / GPU_OUT_DIR / BANK_FILE)
    with pytest.raises(RuntimeError, match="another index"):
        build_index.run(
            config,
            "dense",
            settings=settings,
            **{k: v for k, v in _tiny().items() if "encoder" in k and k != "article_encoder"},
        )
    with (index_dir / BM25_HITS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(SystemExit, match=f"changed {BM25_HITS_FILE}"):
        verify.run(config, settings=settings)


def test_retrieval_from_the_built_bank_is_deterministic(built):
    config, settings, _ = built
    paths = rag_paths(config, settings)
    retriever = Retriever.from_file(paths.index(paths.current("index")) / BANK_FILE, top_k=3)
    path = REPO_ROOT / "tests" / "fixtures" / "rag" / "packets" / "packet_long.json"
    packet = json.loads(path.read_text(encoding="utf-8"))
    result = retriever.retrieve(packet)
    assert result.model_dump_json() == retriever.retrieve(packet).model_dump_json()
    assert all(rid.startswith("R") for rids in result.by_driver.values() for rid in rids)
    assert {p.source for p in result.passages} <= {"pubmed", "statpearls", "guideline"}


def test_gpu_bundle_collects_the_files(built):
    config, settings, _ = built

    def fake_wheel(folder):
        path = folder / "fate_icu-0.1.0-py3-none-any.whl"
        path.write_bytes(b"wheel")
        return path

    report = gpu_bundle.run(config, settings=settings, wheel_builder=fake_wheel)
    assert set(report["files"]) == {
        "fate_icu-0.1.0-py3-none-any.whl",
        "passages.parquet",
        "gpu_spec.json",
        BM25_HITS_FILE,
    }
    assert "fate_icu.rag.gpu_step --bundle" in report["notebook_commands"][2]
    assert report["notebook_commands"][0].startswith("pip install -q transformers==")


def test_configs_and_guards(tmp_path, monkeypatch):
    full = load_config("configs/rag/full.yaml")
    smoke = load_config("configs/rag/smoke.yaml")
    assert full.require_outside_onedrive and not smoke.require_outside_onedrive
    assert [g.id for g in full.sources.guidelines.items] == [
        "ssc2021",
        "kdigo_aki_2012",
        "esicm_ards_2023",
        "padis_2018",
    ]
    settings = _settings(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="confirm the licence of ssc2021, kdigo_aki_2012"):
        build_corpus._guidelines(full, rag_paths(full, settings))
    monkeypatch.setenv("OneDrive", str(tmp_path))
    with pytest.raises(RuntimeError, match="outside OneDrive"):
        rag_paths(full, settings)
    assert rag_paths(smoke, settings).build == tmp_path / "rag" / "rag_smoke"  # small: allowed
    with pytest.raises(ValidationError, match="pool_depth"):
        RagConfig.model_validate(
            {**smoke.model_dump(), "evaluation": {**smoke.evaluation.model_dump(), "pool_depth": 9}}
        )


def test_index_version_covers_corpus_bm25_and_encoders():
    config = load_config("configs/rag/smoke.yaml")
    base = index_version("a" * 64, config)
    assert base == index_version("a" * 64, config) and len(base) == 16
    assert index_version("b" * 64, config) != base
    assert index_version("a" * 64, config.model_copy(update={"bm25": BM25Params(k1=1.2)})) != base
    other = config.models.model_copy(update={"article_encoder": ModelRef(repo="ncbi/x", revision="1" * 40)})
    assert index_version("a" * 64, config.model_copy(update={"models": other})) != base
    rerank_only = config.retrieval.model_copy(update={"rerank_depth": 9})
    assert (
        index_version("a" * 64, config.model_copy(update={"retrieval": rerank_only})) == base
    )  # bank, not index


def test_fetch_needs_the_ncbi_email(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="NCBI_EMAIL"):
        fetch.eutils_client(settings)
    monkeypatch.setenv("NCBI_EMAIL", "someone@example.org")
    client = fetch.eutils_client(load_settings(env_file=tmp_path / "none.env"))
    assert "someone@example.org" not in repr(client)
    assert fetch.run(load_config("configs/rag/smoke.yaml"), "pubmed", settings=settings)["pubmed"].startswith(
        "uses the fixtures"
    )
