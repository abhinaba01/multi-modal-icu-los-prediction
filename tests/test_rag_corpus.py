"""Stage 5 foundation: passages, the corpus file and its hash, text normalisation, driver keys."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from fate_icu.rag.corpus import (
    PASSAGES_FILE,
    Document,
    Passage,
    Section,
    corpus_sha256,
    load_corpus,
    load_corpus_info,
    passage_id,
    verify_corpus,
    write_corpus,
)
from fate_icu.rag.drivers import WHOLE_IMAGE, driver_keys, image_concepts
from fate_icu.rag.text import normalise_text
from fate_icu.utils.env import load_settings, under_onedrive


def make(doc_id="123", chunk=0, text="Lactate rose.", source="pubmed", **kw) -> Passage:
    return Passage(
        passage_id=passage_id(source, doc_id, chunk),
        source=source,
        doc_id=doc_id,
        chunk=chunk,
        title=kw.pop("title", "A title"),
        text=text,
        n_tokens=kw.pop("n_tokens", 3),
        url="https://example.org",
        licence="test licence",
        citation="Author A. J Test. 2020.",
        **kw,
    )


def test_passage_ids():
    assert passage_id("pubmed", "31234567", 1) == "pubmed:31234567:0001"
    assert passage_id("guideline", "kdigo_aki_2012", 12) == "guideline:kdigo_aki_2012:0012"
    for bad in [("pubmed", "a:b", 0), ("pubmed", "", 0), ("web", "1", 0), ("pubmed", "1", 10_000)]:
        with pytest.raises(ValueError):
            passage_id(*bad)


def test_passage_validation():
    with pytest.raises(ValueError, match="should be"):
        Passage(**{**make().model_dump(), "passage_id": "pubmed:123:0007"})
    with pytest.raises(ValueError, match="no text"):
        make(text="   ")
    with pytest.raises(ValueError):
        Passage(**{**make().model_dump(), "extra": 1})


def test_section_and_document_validation():
    with pytest.raises(ValueError, match="locators"):
        Section("Results", ("a", "b"), ("1",))
    Section("Results", ("a", "b"), ("1", "2"))
    with pytest.raises(ValueError, match="licence"):
        Document("pubmed", "1", "t", (), "u", "", "c")
    with pytest.raises(ValueError, match="doc_id"):
        Document("pubmed", "1 2", "t", (), "u", "l", "c")


def test_search_text_and_encoder_pair():
    p = make(title="Lactic Acidosis", section="Treatment / Management", text="Treat the cause.")
    assert p.search_text() == "Lactic Acidosis\nTreatment / Management\nTreat the cause."
    assert p.encoder_pair() == ("Lactic Acidosis", "Treatment / Management. Treat the cause.")
    assert make(text="Only text.").encoder_pair() == ("A title", "Only text.")


def test_write_read_roundtrip_sorts_and_hashes(tmp_path):
    passages = [make("2", 0, "b"), make("10", 1, "c"), make("10", 0, "a", year=2019, locator="p. 3")]
    info = write_corpus(passages, tmp_path)
    loaded = load_corpus(tmp_path)
    assert [p.passage_id for p in loaded] == sorted(p.passage_id for p in passages)
    assert loaded[0].year == 2019 and loaded[0].locator == "p. 3" and loaded[1].year is None
    assert info == load_corpus_info(tmp_path) == verify_corpus(tmp_path)
    assert info.sha256 == corpus_sha256(loaded)
    assert info.n_passages == 3 and info.n_documents == 2 and info.passages_by_source == {"pubmed": 3}
    # the hash is about content: rewriting the same passages in another order changes nothing
    assert write_corpus(list(reversed(passages)), tmp_path / "again").sha256 == info.sha256


def test_hash_changes_with_any_field():
    base = [make("1", 0, "text")]
    assert corpus_sha256(base) != corpus_sha256([make("1", 0, "text.")])
    assert corpus_sha256(base) != corpus_sha256([make("1", 0, "text", title="Another title")])
    assert corpus_sha256(base) != corpus_sha256([make("1", 0, "text", section="Results")])
    assert corpus_sha256(base) != corpus_sha256([*base, make("2", 0, "text")])


def test_duplicates_and_tampering_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_corpus([make("1", 0), make("1", 0, "other")], tmp_path / "dup")
    write_corpus([make("1", 0, "first"), make("2", 0, "second")], tmp_path)
    table = pq.read_table(tmp_path / PASSAGES_FILE)
    pq.write_table(table.set_column(7, "text", [["first", "changed"]]), tmp_path / PASSAGES_FILE)
    with pytest.raises(RuntimeError, match="differs"):
        verify_corpus(tmp_path)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ﬁbrosis and  edema", "fibrosis and edema"),
        ("hyper­kalaemia", "hyperkalaemia"),
        ("zero​width﻿", "zerowidth"),
        ("line one\nline\ttwo\r\n", "line one line two"),
        ("ctrl\x07char", "ctrl char"),
        ("PaO₂/FiO₂ ≥ 300", "PaO2/FiO2 ≥ 300"),
    ],
)
def test_normalise_text(raw, expected):
    assert normalise_text(raw) == expected


def test_driver_keys_cover_channels_groups_and_concepts(variables, contract):
    keys = driver_keys(variables, contract)
    by_modality = {m: [k.key for k in keys if k.modality == m] for m in ("timeseries", "static", "image")}
    assert by_modality["timeseries"] == variables.names
    assert by_modality["static"] == contract.feature_sets["fate_static"].groups
    assert by_modality["image"] == [*image_concepts(), WHOLE_IMAGE]
    assert len(image_concepts()) == 11 and "Effusion" in image_concepts()
    assert len({(k.modality, k.key) for k in keys}) == len(keys) == 30 + 8 + 12


def test_settings_rag_dir_and_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("FATE_ARTIFACTS_DIR", str(tmp_path / "art"))
    monkeypatch.delenv("FATE_RAG_DIR", raising=False)
    monkeypatch.setenv("NCBI_EMAIL", "someone@example.org")
    monkeypatch.setenv("NCBI_API_KEY", "abc123")
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.rag_dir == tmp_path / "art" / "rag"
    assert settings.ncbi_email.get_secret_value() == "someone@example.org"
    for text in (repr(settings), str(settings), settings.model_dump_json()):
        assert "someone@example.org" not in text and "abc123" not in text
    monkeypatch.setenv("FATE_RAG_DIR", str(tmp_path / "rag"))
    assert load_settings(env_file=tmp_path / "missing.env").rag_dir == tmp_path / "rag"


def test_under_onedrive(tmp_path):
    root = tmp_path / "OneDrive"
    (root / "Desktop").mkdir(parents=True)
    env = {"OneDrive": str(root)}
    assert under_onedrive(root / "Desktop" / "repo", env)
    assert not under_onedrive(tmp_path / "fate-data" / "rag", env)
    assert not under_onedrive(root / "Desktop", {})
