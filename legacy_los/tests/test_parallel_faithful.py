"""Parallel execution must be bit-identical to serial (faithfulness guard).

The models are always built single-threaded inside the workers, so `compute.n_jobs`
only changes scheduling -- never the numbers. This test proves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.evaluation.experiment import (classification_per_repeat, make_splits, regression_per_repeat)
from src.preprocessing.prepare import PreparedData


def _prepared(n=80, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n, 6), columns=[f"f{i}" for i in range(6)],
                     index=pd.Index(range(n), name="stay_id"))
    los = np.clip(2.0 + (X["f0"] + X["f1"]).to_numpy() * 3.0 + rng.rand(n), 1.01, 21.0)
    y_reg = pd.Series(los, index=X.index)
    threshold = float(np.quantile(los, 0.75))
    y_class = pd.Series((los > threshold).astype(int), index=X.index)
    return PreparedData(X=X, y_reg=y_reg, y_class=y_class, los=y_reg,
                        threshold=threshold, feature_columns=list(X.columns))


def _cfg(n_jobs):
    return {
        "project": {"random_state": 666},
        "models": {"rf_n_estimators": 50, "xgb_n_estimators": 50},
        "evaluation": {"n_repeats": 3, "test_size": 0.2},
        "compute": {"n_jobs": n_jobs},
    }


def test_classification_parallel_equals_serial():
    prep = _prepared()
    splits = make_splits(len(prep.X), 3, 0.2, 666)
    serial = classification_per_repeat(prep, _cfg(1), splits)
    parallel = classification_per_repeat(prep, _cfg(2), splits)
    for model in serial:
        pdt.assert_frame_equal(serial[model], parallel[model])


def test_regression_parallel_equals_serial():
    prep = _prepared()
    splits = make_splits(len(prep.X), 3, 0.2, 666)
    serial = regression_per_repeat(prep, _cfg(1), splits)
    parallel = regression_per_repeat(prep, _cfg(2), splits)
    for model in serial:
        pdt.assert_frame_equal(serial[model], parallel[model])
