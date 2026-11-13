"""Stage 5 retrieval evaluation: pool, blind sheet, labels, metrics, statistics, the grading tool."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fate_icu.rag import label as label_tool
from fate_icu.rag.corpus import Passage, passage_id
from fate_icu.rag.evaluate import (
    SHEET_COLUMNS,
    Label,
    blind_order,
    bootstrap_ci,
    build_pool,
    evaluate_systems,
    export_labels,
    format_markdown,
    grades_of,
    holm,
    make_sheet_from_bank,
    ndcg_at_k,
    precision_at_k,
    read_labelling_sheet,
    read_labels,
    score_bank,
    sign_flip_p,
    weighted_kappa,
    write_labelling_sheet,
    write_labels,
)
from fate_icu.rag.retrieve import SYSTEMS, BankQuery, QueryBank, RankedHit, save_query_bank

RANKINGS = {
    "q1": {
        "bm25": ["a", "b", "c"],
        "dense": ["c", "d"],
        "hybrid": ["c", "a", "d"],
        "hybrid_rerank": ["d", "c", "a"],
    },
    "q2": {"bm25": ["e"], "dense": ["f", "e"], "hybrid": ["e", "f"], "hybrid_rerank": ["f", "e"]},
}


def _passages(ids):
    return {
        i: SimpleNamespace(source="pubmed", title=f"T {i}", section="", locator="", text=f"text\t{i}\nmore")
        for i in ids
    }


def test_pool_and_blind_order():
    pool = build_pool(RANKINGS, ["q1", "q2"], depth=2)
    assert pool == {"q1": ["a", "b", "c", "d"], "q2": ["e", "f"]}
    reversed_pool = build_pool(RANKINGS, ["q1", "q2"], systems=SYSTEMS[::-1], depth=2)
    assert blind_order(pool, 1) == blind_order(reversed_pool, 1)  # the order hides the system
    assert sorted(blind_order(pool, 1)["q1"]) == sorted(pool["q1"])
    assert blind_order(pool, 1) == blind_order(pool, 1) and blind_order(pool, 1) != blind_order(pool, 2)
    with pytest.raises(KeyError):
        build_pool(RANKINGS, ["q3"])
    with pytest.raises(KeyError, match="no tfidf"):
        build_pool(RANKINGS, ["q1"], systems=["tfidf"])


def test_sheet_round_trips_in_every_excel_encoding(tmp_path):
    path = tmp_path / "sheet.tsv"
    n = write_labelling_sheet(path, {"q1": ["a", "b"]}, {"q1": "query one"}, _passages("ab"), "v1")
    assert n == 2 and path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = read_labelling_sheet(path)
    assert (
        [r.passage_id for r in rows] == ["a", "b"] and rows[0].text == "text a more" and rows[0].grade is None
    )
    text = path.read_bytes().decode("utf-8-sig").replace("\t\t\r\n", "\t1.0\t\r\n", 1)  # Excel writes 1.0
    assert "\t1.0\t" in text
    for encoding, data in (("utf-16", text.encode("utf-16")), ("utf-8 no BOM", text.encode("utf-8"))):
        path.write_bytes(data)
        assert [r.grade for r in read_labelling_sheet(path)] == [1, None], encoding
    path.write_text(text.replace("\t1.0\t", "\t3\t"), encoding="utf-8")
    with pytest.raises(ValueError, match="not 0, 1 or 2"):
        read_labelling_sheet(path)
    path.write_text("\t".join(SHEET_COLUMNS[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected columns"):
        read_labelling_sheet(path)


def test_export_and_merge_labels(tmp_path):
    sheet = tmp_path / "sheet.tsv"
    write_labelling_sheet(sheet, {"q1": ["a", "b"]}, {"q1": "q"}, _passages("ab"), "v1")
    rows = read_labelling_sheet(sheet)
    with pytest.raises(ValueError, match="2 of 2 rows are not graded"):
        export_labels(rows, "AM")
    assert export_labels(rows, "AM", allow_partial=True) == []
    graded = [r.__class__(**{**r.__dict__, "grade": g}) for r, g in zip(rows, (2, 0), strict=True)]
    with pytest.raises(ValueError, match="rater"):
        export_labels(graded, " ")
    labels_path = tmp_path / "relevance_labels.tsv"
    assert write_labels(labels_path, export_labels(graded, "AM")) == 2
    assert write_labels(labels_path, [Label("q1", "a", 1, "MD", "v1")]) == 3
    assert write_labels(labels_path, [Label("q1", "b", 1, "AM", "v1")]) == 2  # AM's rows replaced
    assert read_labels(labels_path) == [Label("q1", "b", 1, "AM", "v1"), Label("q1", "a", 1, "MD", "v1")]
    assert "text" not in labels_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="choose one"):
        grades_of(read_labels(labels_path))
    assert grades_of(read_labels(labels_path), "MD") == {("q1", "a"): 1}


GRADES = {"a": 2, "b": 0, "c": 1, "d": 2}


def test_precision_hand_computed():
    assert precision_at_k(["a", "b", "c"], GRADES, 3, 1) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "b", "c"], GRADES, 3, 2) == pytest.approx(1 / 3)
    assert precision_at_k(["a"], GRADES, 3, 1) == pytest.approx(1 / 3)  # the denominator stays k
    with pytest.raises(ValueError, match="no grade"):
        precision_at_k(["z"], GRADES, 3, 1)
    assert precision_at_k(["z"], GRADES, 3, 1, unjudged="zero") == 0


def test_ndcg_hand_computed():
    dcg = 3 / math.log2(3) + 1 / math.log2(4)  # gains 0, 3, 1 at ranks 1-3
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)  # 2, 2, 1, 0
    assert ndcg_at_k(["b", "a", "c"], GRADES, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "d", "c", "b"], GRADES, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["b"], {"b": 0}, 5) == 0.0  # no relevant passage


def test_statistics():
    assert bootstrap_ci([0.5, 0.5, 0.5], 200, 1) == (0.5, 0.5)
    values = np.random.default_rng(0).random(20)
    assert bootstrap_ci(values, 500, 3) == bootstrap_ci(values, 500, 3)
    assert sign_flip_p([1, 1, 1], 10_000, 0) == pytest.approx(2 / 8)
    assert sign_flip_p([0, 0, 0], 10_000, 0) == 1.0
    diff = np.random.default_rng(1).normal(0.3, 1, size=12)
    exact = sign_flip_p(diff, 10_000, 0)  # 4096 sign patterns: exact
    assert sign_flip_p(diff, 4_000, 0) == pytest.approx(exact, abs=0.03)  # Monte Carlo
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1]) == pytest.approx(1 - 0.125 / 0.34375)


def test_evaluate_systems_and_markdown():
    grades = {("q1", "a"): 2, ("q1", "b"): 0, ("q1", "c"): 1, ("q1", "d"): 0, ("q2", "e"): 0, ("q2", "f"): 0}
    result = evaluate_systems(RANKINGS, grades, ["q1", "q2"], n_boot=200, n_perm=1000)
    assert result["queries_without_relevant_passage"] == 1 and result["judged_passages"] == 6
    assert result["per_query"]["bm25"]["P@3"] == pytest.approx([2 / 3, 0.0])
    assert result["summary"]["bm25"]["P@3"]["mean"] == pytest.approx(1 / 3)
    assert result["per_query"]["hybrid_rerank"]["P@3 strict"] == pytest.approx([1 / 3, 0.0])
    for name in result["metrics"]:
        rows = result["comparisons"][name]
        assert [r["comparison"] for r in rows] == [
            "hybrid vs bm25",
            "hybrid vs dense",
            "hybrid_rerank vs hybrid",
            "dense vs bm25",
        ]
        assert all(r["p_holm"] >= r["p"] for r in rows)
    text = format_markdown(result)
    assert "| hybrid_rerank |" in text and "Holm p" in text and "1 queries without a relevant passage" in text
    with pytest.raises(ValueError, match="no grade"):
        evaluate_systems(RANKINGS, {("q1", "a"): 2}, ["q1"])


def _bank(tmp_path):
    def p(i):
        return Passage(
            passage_id=passage_id("pubmed", i, 0),
            source="pubmed",
            doc_id=i,
            chunk=0,
            title=f"T{i}",
            text=f"Passage {i}.",
            n_tokens=2,
            url="u",
            licence="l",
            citation="c",
        )

    passages = {x.passage_id: x for x in (p(str(i)) for i in range(1, 7))}
    ids = list(passages)
    queries = {}
    for direction in ("longer", "shorter"):
        qid = f"timeseries:lactate:{direction}"
        hits = {
            s: [RankedHit(passage_id=pid, score=1.0) for pid in ids[n : n + 3]] for n, s in enumerate(SYSTEMS)
        }
        queries[qid] = BankQuery(
            query_id=qid,
            modality="timeseries",
            key="lactate",
            direction=direction,
            text=f"lactate associated with {direction} intensive care unit length of stay",
            rankings=hits,
        )
    bank = QueryBank(
        index_version="v9",
        corpus_sha256="c" * 64,
        query_spec_sha256="q" * 64,
        template="t",
        params={"store_depth": 3},
        queries=queries,
        passages=passages,
    )
    return save_query_bank(bank, tmp_path / "query_bank.json")


def test_sheet_import_and_score_from_a_bank(tmp_path):
    bank_path = _bank(tmp_path)
    report = make_sheet_from_bank(bank_path, tmp_path / "eval" / "sheet.tsv", depth=3)
    assert report == {
        "sheet": str(tmp_path / "eval" / "sheet.tsv"),
        "queries": 1,
        "rows": 6,
        "index_version": "v9",
    }
    rows = read_labelling_sheet(tmp_path / "eval" / "sheet.tsv")
    assert {r.query_id for r in rows} == {
        "timeseries:lactate:longer"
    }  # only the "longer" queries are labelled
    graded = [r.__class__(**{**r.__dict__, "grade": int(r.passage_id[-6]) % 3}) for r in rows]
    write_labels(tmp_path / "labels.tsv", export_labels(graded, "AM"))
    result = score_bank(bank_path, tmp_path / "labels.tsv", tmp_path / "out", n_boot=100, n_perm=100)
    assert result["queries"] == 1 and result["index_version"] == "v9" and result["rater"] == "AM"
    assert (tmp_path / "out" / "metrics.json").exists() and (tmp_path / "out" / "metrics.md").exists()
    write_labels(tmp_path / "labels.tsv", [Label(r.query_id, r.passage_id, 1, "AM", "v8") for r in rows])
    with pytest.raises(RuntimeError, match="for index"):
        score_bank(bank_path, tmp_path / "labels.tsv", tmp_path / "out")


def test_labelling_tool_with_scripted_answers(tmp_path):
    sheet = tmp_path / "sheet.tsv"
    write_labelling_sheet(sheet, {"q1": ["a", "b", "c"]}, {"q1": "query one"}, _passages("abc"), "v1")
    answers = iter(["2", "x", "h", "s", "1", "b", "0", "q"])
    said = []
    summary = label_tool.run(sheet, ask=lambda prompt: next(answers), say=said.append)
    assert summary == {"graded": 2, "total": 3}
    assert [r.grade for r in read_labelling_sheet(sheet)] == [2, None, 0]  # "b" went back and regraded c
    assert any("Please type" in line for line in said) and any("directly relevant" in line for line in said)
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")
    again = label_tool.run(sheet, ask=lambda prompt: "1", say=said.append)  # resumes at the ungraded row
    assert again == {"graded": 3, "total": 3} and [r.grade for r in read_labelling_sheet(sheet)] == [2, 1, 0]
    assert label_tool.run(sheet, ask=lambda prompt: "q", say=said.append) == {"graded": 3, "total": 3}


def test_labelling_tool_survives_a_console_that_cannot_show_a_character(tmp_path, monkeypatch):
    import io

    sheet = tmp_path / "sheet.tsv"
    passages = {
        "a": SimpleNamespace(source="pubmed", title="T", section="", locator="", text="π is not Latin-1")
    }
    write_labelling_sheet(sheet, {"q1": ["a"]}, {"q1": "query"}, passages, "v1")
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", write_through=True)  # a legacy code page
    monkeypatch.setattr("sys.stdout", console)
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    label_tool.main(["--sheet", str(sheet)])
    shown = console.buffer.getvalue().decode("cp1252")
    assert "? is not Latin-1" in shown and "Saved." in shown
