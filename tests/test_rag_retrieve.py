"""Stage 5 retrieve(packet): R ids, deduplication, determinism, and the query-bank file."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fate_icu.rag.corpus import Passage, passage_id
from fate_icu.rag.retrieve import (
    SYSTEMS,
    BankQuery,
    QueryBank,
    RankedHit,
    Retriever,
    load_query_bank,
    passages_for_prompt,
    retrieve,
    save_query_bank,
)
from fate_icu.utils.env import REPO_ROOT


def _passage(doc_id, text):
    return Passage(
        passage_id=passage_id("pubmed", doc_id, 0),
        source="pubmed",
        doc_id=doc_id,
        chunk=0,
        title=f"Title {doc_id}",
        text=text,
        n_tokens=5,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/",
        licence="L",
        citation=f"Author {doc_id}. 2020.",
        year=2020,
    )


def _query(modality, key, direction, ids, systems=SYSTEMS):
    query_id = f"{modality}:{key}:{direction}"
    hits = [RankedHit(passage_id=passage_id("pubmed", i, 0), score=float(10 - n)) for n, i in enumerate(ids)]
    return query_id, BankQuery(
        query_id=query_id,
        modality=modality,
        key=key,
        direction=direction,
        text=f"{key} associated with {direction} intensive care unit length of stay",
        rankings={s: hits for s in systems},
    )


@pytest.fixture
def bank():
    passages = {p.passage_id: p for p in (_passage(str(i), f"Text {i}.") for i in range(1, 8))}
    queries = dict(
        [
            _query("timeseries", "creatinine", "longer", ["1", "2", "3", "4"]),
            _query("timeseries", "lactate", "shorter", ["3", "5", "6"]),
            _query("image", "Effusion", "longer", ["2"]),
            _query("static", "sofa_static", "longer", ["7", "1", "5"]),
        ]
    )
    return QueryBank(
        index_version="v1",
        corpus_sha256="c" * 64,
        query_spec_sha256="q" * 64,
        template="{label} associated with {direction} intensive care unit length of stay",
        params={"store_depth": 4},
        queries=queries,
        passages=passages,
    )


PACKET = {
    "prediction": {"p_long_stay": 0.78},
    "drivers": [
        {
            "id": "D1",
            "modality": "timeseries",
            "key": "creatinine",
            "label": "serum creatinine 2.3 mg/dL",
            "observed": "rose from 1.1 to 2.3 mg/dL",
            "direction": "longer",
            "effect": 0.42,
        },
        {
            "id": "D2",
            "modality": "image",
            "key": "Effusion",
            "label": "pleural effusion (estimate 0.81)",
            "direction": "longer",
            "effect": 0.21,
        },
        {"id": "D3", "modality": "static", "key": "sofa_static", "direction": "longer", "effect": 0.1},
    ],
}


def test_r_ids_dedupe_and_driver_lists(bank):
    result = Retriever(bank, top_k=3).retrieve(PACKET)
    assert result.by_driver == {"D1": ("R1", "R2", "R3"), "D2": ("R2",), "D3": ("R4", "R1", "R5")}
    assert [p.passage_id for p in result.passages] == [
        passage_id("pubmed", i, 0) for i in ["1", "2", "3", "7", "5"]
    ]
    assert result.passages[0].drivers == ("D1", "D3") and result.passages[1].drivers == ("D1", "D2")
    assert [p.rid for p in result.passages] == ["R1", "R2", "R3", "R4", "R5"]
    assert result.queries["D1"] == "creatinine associated with longer intensive care unit length of stay"


def test_patient_fields_never_reach_retrieval(bank):
    result = Retriever(bank).retrieve(PACKET)
    dumped = result.model_dump_json()
    for value in ("2.3", "mg/dL", "0.81", "rose from"):
        assert value not in dumped
    changed = json.loads(json.dumps(PACKET))
    changed["drivers"][0]["label"] = "something else entirely"
    changed["drivers"][0]["observed"] = "fell"
    assert Retriever(bank).retrieve(changed) == result  # only (modality, key, direction) and order matter


def test_deterministic_and_order_dependent(bank):
    retriever = Retriever(bank)
    assert retriever.retrieve(PACKET).model_dump_json() == retriever.retrieve(PACKET).model_dump_json()
    flipped = {"drivers": list(reversed(PACKET["drivers"]))}
    assert retriever.retrieve(flipped).by_driver["D3"] == ("R1", "R2", "R3")


def test_packet_forms_top_k_and_shared_keys(bank):
    objects = SimpleNamespace(
        drivers=[
            SimpleNamespace(id="D1", modality="timeseries", key="lactate", direction="shorter"),
            SimpleNamespace(id="D2", modality="timeseries", key="lactate", direction="shorter"),
        ]
    )
    result = Retriever(bank, top_k=2).retrieve(objects)
    assert result.by_driver == {"D1": ("R1", "R2"), "D2": ("R1", "R2")} and len(result.passages) == 2
    fewer = Retriever(bank, top_k=4).retrieve({"drivers": [PACKET["drivers"][1]]})
    assert fewer.by_driver == {"D2": ("R1",)}  # the bank holds a single hit for this query
    assert Retriever(bank).retrieve({"drivers": []}).passages == ()


def test_loud_failures(bank):
    retriever = Retriever(bank)
    with pytest.raises(KeyError, match="timeseries:creatinine:shorter"):
        retriever.retrieve(
            {"drivers": [{"id": "D1", "modality": "timeseries", "key": "creatinine", "direction": "shorter"}]}
        )
    with pytest.raises(ValueError, match="duplicate driver ids"):
        retriever.retrieve({"drivers": [PACKET["drivers"][0], PACKET["drivers"][0]]})
    with pytest.raises(ValueError, match="no drivers"):
        retriever.retrieve({"prediction": {}})
    with pytest.raises(ValidationError):
        retriever.retrieve(
            {"drivers": [{"id": "X1", "modality": "timeseries", "key": "lactate", "direction": "longer"}]}
        )
    with pytest.raises(ValueError, match="system"):
        Retriever(bank, system="tfidf")
    with pytest.raises(ValueError, match="stored depth"):
        Retriever(bank, top_k=5)
    empty_id, empty = _query("timeseries", "sodium", "longer", [])
    bank_with_empty = bank.model_copy(update={"queries": {**bank.queries, empty_id: empty}})
    with pytest.raises(ValueError, match="no hybrid_rerank results"):
        Retriever(bank_with_empty).retrieve(
            {"drivers": [{"id": "D1", "modality": "timeseries", "key": "sodium", "direction": "longer"}]}
        )


def test_bank_validation(bank):
    data = bank.model_dump()
    data["passages"].pop(passage_id("pubmed", "2", 0))
    with pytest.raises(ValidationError, match="not stored"):
        QueryBank.model_validate(data)
    one_system = dict([_query("timeseries", "x", "longer", ["1"], systems=("bm25",))])
    with pytest.raises(ValidationError, match="rankings for"):
        QueryBank.model_validate({**bank.model_dump(), "queries": one_system})


def test_save_load_and_hash(tmp_path, bank):
    path = save_query_bank(bank, tmp_path / "query_bank.json")
    loaded = load_query_bank(path)
    assert loaded == bank and loaded.sha256 == bank.sha256
    assert save_query_bank(loaded, tmp_path / "again.json").read_bytes() == path.read_bytes()
    changed = bank.model_copy(update={"index_version": "v2"})
    assert changed.sha256 != bank.sha256


def test_prompt_view_and_convenience(bank):
    result = retrieve(PACKET, Retriever(bank))
    view = passages_for_prompt(result)
    assert view[0] == {
        "rid": "R1",
        "citation": "Author 1. 2020.",
        "title": "Title 1",
        "section": "",
        "locator": "",
        "text": "Text 1.",
    }


def test_example_packets_parse(bank):
    for path in sorted((REPO_ROOT / "tests" / "fixtures" / "rag" / "packets").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert packet["drivers"] and all("key" in d for d in packet["drivers"])
