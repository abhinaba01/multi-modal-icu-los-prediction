"""The planted benchmark data: the planted units are really what the label depends on."""

from __future__ import annotations

import numpy as np
import pytest

from fate_icu.attribution.synthetic import CAUSES, planted_stays


@pytest.fixture(scope="module")
def planted():
    return planted_stays(n=900, seed=1)


def test_truth_marks_exactly_the_active_causes(planted):
    space, truth, causes = planted.space, planted.truth, planted.causes
    assert truth.sum(axis=1).tolist() == causes.sum(axis=1).tolist()
    assert planted.space.units[space.ts_unit("c2", 2)].block == "hours 8-12"
    assert (
        truth[causes[:, 0], space.ts_unit("c2", 2)].all()
        and not truth[~causes[:, 0], space.ts_unit("c2", 2)].any()
    )
    labs = space.n_ts + list(planted.data.static_groups).index("labs")
    assert truth[causes[:, 2], labs].all()
    assert truth[causes[:, 3], space.concept_slice.start + 1].all()


def test_the_label_follows_the_causes_up_to_a_little_noise(planted):
    long = planted.data.y_long > 0.5
    agree = (long == planted.causes.any(axis=1)).mean()
    assert 0.95 < agree < 1.0
    assert 0.35 < long.mean() < 0.65 and len(CAUSES) == planted.causes.shape[1]


def test_planted_patterns_are_present_where_they_should_be(planted):
    values, causes = planted.data.values, planted.causes
    rise = values[causes[:, 0], 8:12, 2].mean() - values[~causes[:, 0], 8:12, 2].mean()
    fall = values[causes[:, 1], 16:20, 4].mean() - values[~causes[:, 1], 16:20, 4].mean()
    outside = values[causes[:, 0], 0:4, 2].mean() - values[~causes[:, 0], 0:4, 2].mean()
    assert rise > 1.5 and fall < -1.5 and abs(outside) < 0.3
    labs = (
        planted.data.static[np.ix_(causes[:, 2], [4, 5])].mean()
        - planted.data.static[np.ix_(~causes[:, 2], [4, 5])].mean()
    )
    assert labs > 1.5


def test_splits_and_determinism(planted):
    assert {"train", "val", "test"} == set(planted.data.split)
    again = planted_stays(n=900, seed=1)
    assert np.array_equal(again.data.values, planted.data.values) and np.array_equal(
        again.truth, planted.truth
    )
    assert not np.array_equal(planted_stays(n=900, seed=2).truth, planted.truth)
