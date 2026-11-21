"""H4 selection comparison and the ablation grid."""

from __future__ import annotations

import numpy as np
import pytest

from fate_icu.eval.grid import BY_ID, GRID, grid_table, runnable_rows
from fate_icu.eval.selection import StaySelection, h4_test, per_stay_values


def linear_stay(rng, n_units=8, noise=1.0):
    """A toy model: logit = bias + sum of the weights of the kept units.

    Removing a unit lowers the logit by its weight, so the best 3 units to select are the 3 largest weights.
    """
    w = rng.uniform(0.0, 2.0, n_units)
    bias = 0.5

    def score_removed(masks):
        return bias + ((~np.asarray(masks, dtype=bool)) * w).sum(axis=1)

    return StaySelection(
        n_units=n_units,
        score_removed=score_removed,
        two_step=np.argsort(-w)[:3],  # the packet's drivers: the truly largest effects
        gradient_scores=w + rng.normal(0, noise, n_units),  # a noisy gradient attribution
    )


def test_per_stay_values_have_the_expected_shapes_and_the_best_selection_is_comprehensive():
    rng = np.random.default_rng(0)
    stays = [linear_stay(rng) for _ in range(60)]
    values = per_stay_values(stays, k=3, seed=1)
    comp = values["comprehensiveness"]
    assert set(comp) == {"two_step", "gradient", "random"} and all(len(v) == 60 for v in comp.values())
    assert comp["two_step"].mean() > comp["gradient"].mean() > comp["random"].mean()
    assert (
        np.all(comp["two_step"] >= comp["random"] - 1e-9) or comp["two_step"].mean() > comp["random"].mean()
    )
    suff = values["sufficiency"]
    assert (
        suff["two_step"].mean() < suff["random"].mean()
    )  # keeping the best units alone keeps the prediction


def test_h4_is_supported_when_two_step_beats_both_and_uses_the_larger_p_value():
    rng = np.random.default_rng(2)
    stays = [linear_stay(rng) for _ in range(150)]
    comp = per_stay_values(stays, k=3, seed=3)["comprehensiveness"]
    out = h4_test(comp, np.arange(150), n_boot=300, seed=4)
    assert out["higher_than_both"] is True
    assert out["p_value"] == max(v["p_value"] for v in out["versus"].values())
    assert out["p_value"] < 0.05


def test_h4_is_not_supported_when_the_selections_are_equally_good():
    rng = np.random.default_rng(5)
    values = rng.random(120)
    same = {"two_step": values, "gradient": values.copy(), "random": values.copy()}
    out = h4_test(same, np.arange(120), n_boot=200, seed=1)
    assert out["higher_than_both"] is False and out["p_value"] == 1.0


def test_the_grid_has_every_row_of_the_spec_once():
    assert [r.id for r in GRID] == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "C1", "C2"]
    assert BY_ID["E7"].rag and BY_ID["E8"].text == "sft_dpo" and BY_ID["C1"].control and not BY_ID["E5"].rag
    assert (
        BY_ID["C1"].text == BY_ID["E5"].text == "zero_shot"
    )  # H2 isolates the input: same model, same prompt
    assert len(grid_table()) == 11


def test_a_partial_run_reports_the_rows_it_can_and_what_the_others_lack():
    ok, lacking = runnable_rows(
        arms={"B0", "B3"}, packets={"attribution", "shap"}, texts={"E4"}, has_passages=False
    )
    assert [r.id for r in ok] == ["E0", "E1"]  # E4 has text but its predictor M is missing
    assert lacking["E4"] == ["predictor arm M"]
    assert "retrieved passages" in lacking["E7"] and "generated texts for E7" in lacking["E7"]
    assert lacking["C1"][0] == "predictor arm M"


def test_with_everything_present_every_row_runs():
    ok, lacking = runnable_rows(
        arms={"B0", "B3", "B4", "M"},
        packets={"attribution", "shap", "summary", "random"},
        texts={"E4", "E5", "E6", "E7", "E8", "C1", "C2"},
        has_passages=True,
    )
    assert len(ok) == 11 and lacking == {}


def test_unknown_row_lookup_is_an_error():
    with pytest.raises(KeyError):
        BY_ID["E9"]
