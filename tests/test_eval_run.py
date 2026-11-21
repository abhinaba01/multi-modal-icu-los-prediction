"""The Stage 7 orchestrator end to end, on the synthetic bundle (offline checker, small bootstrap)."""

from __future__ import annotations

import json

import pytest

from fate_icu.eval import explanation as X
from fate_icu.eval import grounding as G
from fate_icu.eval.run import EvalConfig, evaluate, load_config, run, write_report
from fate_icu.eval.smoke_data import synthetic_bundle

CONFIG = EvalConfig(run_name="eval_test", n_stays=160, n_boot=400, seed=7, figures=True)


@pytest.fixture(scope="module")
def results():
    lexicon = X.build_lexicon(concepts=["Effusion", "Edema", "Atelectasis"])
    return evaluate(synthetic_bundle(CONFIG.n_stays, CONFIG.seed), CONFIG, G.LexicalChecker(), lexicon)


def test_every_registered_hypothesis_is_tested_on_the_full_synthetic_bundle(results):
    tests = results["hypotheses"]["tests"]
    assert set(tests) == {"H1", "H2", "H3", "H4", "H5"} and results["hypotheses"]["not_run"] == {}
    for t in tests.values():
        assert 0 < t["p_value"] <= 1 and t["holm"]["p_adjusted"] >= t["holm"]["p"]
        assert isinstance(t["supported"], bool)


def test_the_planted_differences_are_found(results):
    tests = results["hypotheses"]["tests"]
    assert tests["H2"]["difference"] < 0  # the packet-constrained text names fewer unattributed units than C1
    assert tests["H3"]["difference"] > 0  # E7 cites its passages, E6 does not
    assert tests["H4"]["higher_than_both"] is True
    assert tests["H2"]["supported"] and tests["H3"]["supported"] and tests["H4"]["supported"]


def test_the_report_holds_predictive_faithfulness_and_groundedness_per_row(results):
    assert set(results["predictive"]) == {"B0", "B3", "B4", "M"}
    ex = results["explanation"]
    assert set(ex) == {"E4", "E5", "E6", "E7", "E8", "C1", "C2"}
    assert ex["E4"]["unattributed_rate"]["mean"] == 0.0 and ex["E4"]["numeric_consistency"]["mean"] == 1.0
    assert ex["C1"]["unattributed_rate"]["mean"] > ex["E5"]["unattributed_rate"]["mean"]
    assert (
        results["grounding"]["E7"]["citation_recall"]["ratio"]
        > results["grounding"]["E6"]["citation_recall"]["ratio"]
    )
    assert results["grid"]["lacking"] == {} and len(results["grid"]["runnable"]) == 11


def test_a_missing_row_makes_its_hypothesis_not_run_with_a_reason_and_never_a_number():
    bundle = synthetic_bundle(120, 3)
    del bundle.explained["E8"], bundle.explained["C1"]
    del bundle.arms["M"]
    out = evaluate(bundle, CONFIG, G.LexicalChecker(), X.build_lexicon(concepts=["Effusion"]))
    assert set(out["hypotheses"]["not_run"]) == {"H1", "H2", "H5"}
    assert "M and B3" in out["hypotheses"]["not_run"]["H1"] and "DPO" in out["hypotheses"]["not_run"]["H5"]
    assert set(out["hypotheses"]["tests"]) == {"H3", "H4"}
    assert "E8" in out["grid"]["lacking"] and "C1" in out["grid"]["lacking"]


def test_the_report_writes_tables_json_manifest_and_a_figure(results, tmp_path):
    out = write_report(results, tmp_path / "eval_test_x", CONFIG)
    for name in ("predictive", "explanation_faithfulness", "groundedness", "hypotheses", "grid"):
        assert (out / "tables" / f"{name}.csv").exists() and (out / "tables" / f"{name}.md").exists()
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(saved["hypotheses"]["tests"]) == {"H1", "H2", "H3", "H4", "H5"}
    assert (out / "run_manifest.json").exists() and (out / "unattributed_rate.png").exists()
    table = (out / "tables" / "hypotheses.md").read_text(encoding="utf-8")
    assert table.startswith("| hypothesis |") and "H2" in table


def test_run_is_deterministic_for_a_fixed_seed(tmp_path):
    a = run(CONFIG.model_copy(update={"figures": False, "n_stays": 100}), tmp_path / "a")
    b = run(CONFIG.model_copy(update={"figures": False, "n_stays": 100}), tmp_path / "b")
    ja, jb = (json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in (a, b))
    assert ja == jb


def test_the_full_source_stops_with_a_message_instead_of_guessing(tmp_path):
    with pytest.raises(SystemExit, match="Stage 6"):
        run(CONFIG.model_copy(update={"source": "vm"}), tmp_path)


def test_the_shipped_configs_load():
    smoke = load_config("configs/eval/smoke.yaml")
    full = load_config("configs/eval/full.yaml")
    assert smoke.source == "synthetic" and smoke.checker == "lexical"
    assert full.source == "vm" and full.checker == "nli" and full.n_boot == 1000
