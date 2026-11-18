"""Stage 4 driver selection, with an additive stand-in model so every number is known."""

from __future__ import annotations

import numpy as np
import pytest
from stay_data import random_stay_data

from fate_icu.attribution.gates import build_layout
from fate_icu.attribution.select import (
    SelectionConfig,
    _merge_runs,
    candidate_units,
    fit_tau,
    modality_slices,
    select_drivers,
)
from fate_icu.attribution.units import UnitSpace

WIDTH, CONCEPTS = 9, ["c_a", "c_b", "c_c"]


@pytest.fixture(scope="module")
def space():
    data = random_stay_data(n=60, seed=5, image_share=0.9)
    return UnitSpace(build_layout(data, CONCEPTS, feature_width=WIDTH))


class Additive:
    """A stay whose logit is ``base + sum of the effects of the kept units``."""

    def __init__(self, space, effects, base=0.0):
        self.space, self.effects, self.base = space, np.asarray(effects, dtype=float), base

    def logits(self, removed):
        removed = np.atleast_2d(removed)
        return self.base + ((~removed) * self.effects).sum(axis=1)

    def base_logit(self):
        return float(self.logits(np.zeros((1, len(self.space)), dtype=bool))[0])

    def removal_delta(self, units=None):
        units = np.arange(len(self.space)) if units is None else np.asarray(units)
        return self.effects[units].copy()


def test_the_unit_space_orders_time_series_then_static_then_image(space):
    assert len(space.blocks) == 6 and space.blocks[0] == (0, 4) and space.blocks[-1] == (20, 24)
    assert space.n_ts == 5 * 6 and space.n_static == 3 and space.n_concepts == 3
    assert len(space) == 30 + 3 + 3 + 1 and space.units[-1].key == "whole_image"
    unit = space.units[space.ts_unit("c2", 2)]
    assert (unit.modality, unit.key, unit.start_hour, unit.end_hour) == ("timeseries", "c2", 8, 12)
    assert unit.block == "hours 8-12"
    slices = modality_slices(space)
    assert [len(v) for v in slices.values()] == [30, 3, 4]  # three concepts and the whole image


def test_a_block_length_that_does_not_divide_the_window_gives_a_short_last_block():
    data = random_stay_data(n=60, seed=5)
    odd = UnitSpace(build_layout(data, CONCEPTS, feature_width=WIDTH), block_hours=5)
    assert odd.blocks[-1] == (20, 24) and len(odd.blocks) == 5


def test_candidates_are_the_largest_scores_of_each_modality(space):
    scores = np.zeros(len(space))
    scores[[0, 1, 2, 3, 4, 5, 6]] = [1, -9, 3, 8, 2, 7, 4]  # time series: 7 nonzero units
    scores[space.n_ts + 1] = -5
    scores[space.concept_slice.start] = 0.5
    chosen = candidate_units(scores, space, per_modality=5)
    ts_part = [u for u in chosen if u < space.n_ts]
    assert sorted(ts_part)[:5] == [1, 2, 3, 5, 6] or set(ts_part) >= {1, 3, 5, 6, 2}
    assert space.n_ts + 1 in chosen and space.concept_slice.start in chosen
    assert len([u for u in chosen if u < space.n_ts]) == 5  # five per modality at most
    assert len([u for u in chosen if space.static_slice.start <= u < space.static_slice.stop]) == 3


def test_merge_joins_adjacent_blocks_of_one_variable_that_push_the_same_way(space):
    deltas = {1: 0.4, 2: 0.3, 3: -0.2, 6: 0.5, 7: 0.1, 5: 0.2, 40: 0.2}
    groups = _merge_runs(sorted(deltas), deltas, space)
    assert [1, 2] in groups  # same variable, adjacent, same sign
    assert [3] in groups  # opposite sign: not merged
    assert [6, 7] in groups
    # unit 5 is the last block of variable 0 and unit 6 the first of variable 1: never merged
    assert [5] in groups and [40] in groups


def test_tau_is_the_quantile_of_random_non_candidate_deltas(space):
    effects = np.zeros(len(space))
    effects[: space.n_ts] = np.linspace(0.0, 1.0, space.n_ts)
    scorer = Additive(space, effects)
    candidate = np.arange(space.n_ts - 5, space.n_ts)  # the five biggest effects are candidates
    cfg = SelectionConfig(n_random_units=10_000, tau_quantile=0.95)
    tau = fit_tau([scorer], [candidate], cfg, seed=0)
    pool = np.abs(np.delete(effects, candidate))
    assert tau == pytest.approx(np.quantile(pool, 0.95))
    assert tau < effects.max()  # a candidate with a big effect passes it
    with pytest.raises(ValueError, match="no validation stay"):
        fit_tau([], [], cfg)


def test_drivers_are_gated_by_tau_sorted_by_effect_and_capped(space):
    effects = np.zeros(len(space))
    big = [space.ts_unit("c0", 0), space.ts_unit("c1", 3), space.ts_unit("c2", 5)]
    effects[big] = [0.9, -1.2, 0.5]
    effects[space.n_ts + 2] = 0.8  # a static group
    effects[space.concept_slice.start + 1] = -0.3
    scorer = Additive(space, effects)
    scores = np.abs(effects) + 1e-3  # a perfect nomination
    drivers = select_drivers(scorer, scores, tau=0.4, cfg=SelectionConfig())
    assert [round(abs(d.delta), 2) for d in drivers] == [1.2, 0.9, 0.8, 0.5]  # 0.3 is below tau
    assert [d.direction for d in drivers] == ["shorter", "longer", "longer", "longer"]
    assert [d.modality for d in drivers] == ["timeseries", "timeseries", "static", "timeseries"]
    capped = select_drivers(scorer, scores, tau=0.0, cfg=SelectionConfig(max_drivers=2))
    assert len(capped) == 2 and abs(capped[0].delta) >= abs(capped[1].delta)


def test_adjacent_blocks_become_one_driver_with_a_recomputed_effect(space):
    effects = np.zeros(len(space))
    first, second = space.ts_unit("c1", 2), space.ts_unit("c1", 3)
    effects[[first, second]] = [0.6, 0.5]
    scorer = Additive(space, effects)
    [driver] = select_drivers(scorer, np.abs(effects) + 1e-3, tau=0.3)
    assert driver.units == (first, second) and driver.key == "c1"
    assert (driver.start_hour, driver.end_hour) == (8, 16)
    assert driver.delta == pytest.approx(1.1)  # removing both together, recomputed


def test_a_merged_driver_below_tau_is_dropped(space):
    class Interacting(Additive):
        def logits(self, removed):  # removing both blocks together has no effect
            removed = np.atleast_2d(removed)
            out = super().logits(removed)
            both = removed[:, self.pair[0]] & removed[:, self.pair[1]]
            return np.where(both, self.base + self.effects.sum(), out)  # the unremoved logit

    first, second = space.ts_unit("c1", 2), space.ts_unit("c1", 3)
    effects = np.zeros(len(space))
    effects[[first, second]] = [0.6, 0.5]
    scorer = Interacting(space, effects)
    scorer.pair = (first, second)
    assert select_drivers(scorer, np.abs(effects) + 1e-3, tau=0.3) == []


def test_units_with_no_effect_are_never_drivers_even_at_tau_zero(space):
    scorer = Additive(space, np.zeros(len(space)))
    assert select_drivers(scorer, np.ones(len(space)), tau=0.0) == []
