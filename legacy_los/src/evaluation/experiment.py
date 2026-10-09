"""Repeated-split experiment runner producing the paper's Tables 3-6.

All experiments use 10 repeated random 80/20 splits and report the mean, matching
the paper. Splits are deterministic given the dataset size and base seed, so the
same rows land in train/test across feature sets -- which is what makes the
ablation a paired comparison.

* :func:`run_classification` -> Table 3 (binary short/long, 0.5 threshold).
* :func:`run_regression`     -> Table 4 (full LOS 1-21, and actual-short 1-Q3).
* :func:`run_stepwise`       -> Tables 5-6 (RF/XGB prior via G-mean, regress on
  predicted-short).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation.metrics import (CLASSIFICATION_METRICS, REGRESSION_METRICS,
                                     classification_metrics, mean_metrics, regression_metrics)
from src.models.classifiers import CLASSIFIER_NAMES, classifier_factories
from src.models.regressors import REGRESSOR_NAMES, regressor_factories
from src.models.stepwise import confusion_counts, gmean_threshold
from src.preprocessing.prepare import PreparedData

LOGGER = logging.getLogger(__name__)

STEPWISE_PRIORS = ["Random forest", "XGBoost"]


def _n_jobs(config: dict[str, Any]) -> int:
    """Worker count for the parallel repeat/model loops.

    Reads ``compute.n_jobs`` (default -1 = all available cores). The same value
    works on any machine -- joblib detects the actual core count at runtime, so no
    code change is needed between a 12-core and a 24-core host.

    Args:
        config: Parsed configuration.

    Returns:
        Number of parallel workers (-1 = all cores).
    """
    return int(config.get("compute", {}).get("n_jobs", -1))


def make_splits(n: int, n_repeats: int, test_size: float, base_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic repeated 80/20 positional splits.

    Args:
        n: Number of rows.
        n_repeats: Number of repeats.
        test_size: Test fraction (0.20).
        base_seed: Base random seed; repeat ``r`` uses ``base_seed + r``.

    Returns:
        List of ``(train_positions, test_positions)`` arrays.
    """
    n_train = int(np.floor((1.0 - test_size) * n))
    splits = []
    for r in range(n_repeats):
        perm = np.random.RandomState(base_seed + r).permutation(n)
        splits.append((perm[:n_train], perm[n_train:]))
    return splits


def _positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (long), robust across estimators.

    Args:
        model: A fitted classifier.
        X: Feature matrix.

    Returns:
        1-D array of positive-class scores.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


# Models that are single-threaded per fit (libsvm) -> parallelize ACROSS repeats.
# Tree models (RF/XGB) are NOT here: they use all cores within one fit, which is
# far more efficient than 1 core each, so they run serially with n_jobs=-1.
_PARALLEL_MODELS = {"SVM"}


def _clf_task(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              model_name: str, config: dict[str, Any], model_n_jobs: int) -> dict[str, float]:
    """Fit one classifier on one split and return its metrics.

    Args:
        X: Full (memmapped) feature matrix.
        y: Classification target.
        train: Train row positions.
        test: Test row positions.
        model_name: Classifier name.
        config: Parsed configuration.
        model_n_jobs: Threads for the model (1 when parallelized across tasks; -1
            for the serial tree-model phase).

    Returns:
        Classification metric dict.
    """
    model = classifier_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train], y[train])
    score = _positive_scores(model, X[test])
    pred = model.predict(X[test])
    return classification_metrics(y[test], pred, score)


def _reg_task(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              model_name: str, config: dict[str, Any], model_n_jobs: int) -> dict[str, float]:
    """Fit one regressor on one split and return its metrics.

    Args:
        X: Full (memmapped) feature matrix.
        y: Regression target.
        train: Train row positions.
        test: Test row positions.
        model_name: Regressor name.
        config: Parsed configuration.
        model_n_jobs: Threads for the model.

    Returns:
        Regression metric dict.
    """
    model = regressor_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train], y[train])
    return regression_metrics(y[test], model.predict(X[test]))


def _run_stage_hybrid(X: np.ndarray, y: np.ndarray, splits, model_names: list[str],
                      config: dict[str, Any], task_fn) -> dict[str, list[dict]]:
    """Hybrid scheduler: SVM parallel across repeats; tree/linear serial (all cores).

    The single-threaded SVM fits are run concurrently across cores (a real win
    since they can't use more than one core each). RandomForest / XGBoost run one
    at a time with ``n_jobs=-1`` (their efficient mode -- 1 core each while several
    run at once just saturates memory bandwidth). Results are identical either way.

    Args:
        X: Feature matrix.
        y: Target.
        splits: Repeated-split positions.
        model_names: Models to evaluate.
        config: Parsed configuration.
        task_fn: ``_clf_task`` or ``_reg_task``.

    Returns:
        Mapping ``model -> [metric per repeat]``.
    """
    n = len(splits)
    per_model: dict[str, list[dict]] = {name: [None] * n for name in model_names}
    parallel_names = [m for m in model_names if m in _PARALLEL_MODELS]
    serial_names = [m for m in model_names if m not in _PARALLEL_MODELS]

    if parallel_names:
        tasks = [(r, name) for r in range(n) for name in parallel_names]
        results = Parallel(n_jobs=_n_jobs(config), verbose=10)(
            delayed(task_fn)(X, y, splits[r][0], splits[r][1], name, config, 1) for r, name in tasks)
        for (r, name), res in zip(tasks, results):
            per_model[name][r] = res

    for name in serial_names:  # tree/linear models: all cores per fit
        for r, (train, test) in enumerate(splits):
            per_model[name][r] = task_fn(X, y, train, test, name, config, -1)
            LOGGER.info("    %s: repeat %d/%d done", name, r + 1, n)
    return per_model


def _classification_runs(prepared: PreparedData, config: dict[str, Any],
                         splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, list[dict]]:
    """Collect per-repeat classification metrics for every model (hybrid parallel).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        splits: Repeated-split positions.

    Returns:
        Mapping ``model -> [metric_dict per repeat]``.
    """
    X = prepared.X.to_numpy()
    y = prepared.y_class.to_numpy()
    result = _run_stage_hybrid(X, y, splits, CLASSIFIER_NAMES, config, _clf_task)
    LOGGER.info("classification: %d models x %d repeats done", len(CLASSIFIER_NAMES), len(splits))
    return result


def _regression_runs(prepared: PreparedData, config: dict[str, Any], actual_short_only: bool,
                     splits: list[tuple[np.ndarray, np.ndarray]] | None) -> dict[str, list[dict]]:
    """Collect per-repeat regression metrics for every model (hybrid parallel).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        actual_short_only: Restrict to actual short stays (LOS <= Q3).
        splits: Optional splits; recomputed for the (possibly subset) size if None.

    Returns:
        Mapping ``model -> [metric_dict per repeat]``.
    """
    X_df, y_ser, los = prepared.X, prepared.y_reg, prepared.los
    if actual_short_only:
        mask = (los <= prepared.threshold).to_numpy()
        X_df, y_ser = X_df.iloc[mask], y_ser.iloc[mask]
    splits = splits or make_splits(len(X_df), config["evaluation"]["n_repeats"],
                                   config["evaluation"]["test_size"], config["project"]["random_state"])
    X = X_df.to_numpy()
    y = y_ser.to_numpy()
    result = _run_stage_hybrid(X, y, splits, REGRESSOR_NAMES, config, _reg_task)
    LOGGER.info("regression%s: %d models x %d repeats done",
                " (actual-short)" if actual_short_only else "", len(REGRESSOR_NAMES), len(splits))
    return result


def run_classification(prepared: PreparedData, config: dict[str, Any],
                       splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """Train/evaluate the four classifiers over repeated splits (Table 3).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        splits: Optional precomputed splits (for paired ablation).

    Returns:
        DataFrame indexed by model with the mean classification metric columns.
    """
    splits = splits or make_splits(len(prepared.X), config["evaluation"]["n_repeats"],
                                   config["evaluation"]["test_size"], config["project"]["random_state"])
    per_model = _classification_runs(prepared, config, splits)
    table = pd.DataFrame({name: mean_metrics(rows) for name, rows in per_model.items()}).T
    return table[CLASSIFICATION_METRICS]


def run_regression(prepared: PreparedData, config: dict[str, Any], actual_short_only: bool = False,
                   splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """Train/evaluate the four regressors over repeated splits (Table 4).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        actual_short_only: If True, restrict to actual short stays (LOS <= Q3),
            reproducing the parenthetical column of Table 4.
        splits: Optional precomputed splits.

    Returns:
        DataFrame indexed by model with the mean regression metric columns.
    """
    per_model = _regression_runs(prepared, config, actual_short_only, splits)
    table = pd.DataFrame({name: mean_metrics(rows) for name, rows in per_model.items()}).T
    return table[REGRESSION_METRICS]


def classification_per_repeat(prepared: PreparedData, config: dict[str, Any],
                              splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, pd.DataFrame]:
    """Per-repeat classification metrics (for paired ablation significance).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        splits: Shared repeated-split positions.

    Returns:
        Mapping ``model -> DataFrame(rows=repeat, cols=metrics)``.
    """
    per_model = _classification_runs(prepared, config, splits)
    return {name: pd.DataFrame(rows) for name, rows in per_model.items()}


def regression_per_repeat(prepared: PreparedData, config: dict[str, Any],
                          splits: list[tuple[np.ndarray, np.ndarray]],
                          actual_short_only: bool = False) -> dict[str, pd.DataFrame]:
    """Per-repeat regression metrics (for paired ablation significance).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        splits: Shared repeated-split positions.
        actual_short_only: Restrict to actual short stays.

    Returns:
        Mapping ``model -> DataFrame(rows=repeat, cols=metrics)``.
    """
    per_model = _regression_runs(prepared, config, actual_short_only, splits)
    return {name: pd.DataFrame(rows) for name, rows in per_model.items()}


def mean_table(per_repeat: dict[str, pd.DataFrame], model_order: list[str], metrics: list[str]) -> pd.DataFrame:
    """Average per-repeat metrics into a mean table (index = model).

    Args:
        per_repeat: Mapping ``model -> DataFrame(rows=repeat, cols=metrics)``.
        model_order: Row order for the output.
        metrics: Column order for the output.

    Returns:
        DataFrame indexed by model with the mean of each metric.
    """
    rows = {name: per_repeat[name].mean() for name in model_order if name in per_repeat}
    return pd.DataFrame(rows).T[metrics]


def _prior_short_indices(X, yc, train, test, prior, config):
    """Fit a prior classifier and return its confusion + predicted-short row ids.

    Args:
        X: Feature matrix.
        yc: Classification target.
        train: Train row positions.
        test: Test row positions.
        prior: Prior classifier name (RF/XGB) -- run with all cores.
        config: Parsed configuration.

    Returns:
        ``(confusion_dict, train_short_ids, test_short_ids)`` where the id arrays
        are ``None`` if there are too few predicted-short rows to regress.
    """
    clf = classifier_factories(config, n_jobs=-1)[prior]()
    clf.fit(X[train], yc[train])
    score_tr = _positive_scores(clf, X[train])
    score_te = _positive_scores(clf, X[test])
    threshold = gmean_threshold(yc[train], score_tr)
    conf = confusion_counts(yc[test], score_te >= threshold)
    train_short = train[score_tr < threshold]
    test_short = test[score_te < threshold]
    if len(train_short) < 2 or len(test_short) < 2:
        return conf, None, None
    return conf, train_short, test_short


def _reg_on_idx(X, yr, train_idx, test_idx, model_name, config, model_n_jobs):
    """Fit a regressor on explicit train/test row ids (predicted-short subset).

    Args:
        X: Feature matrix.
        yr: Regression target.
        train_idx: Train row ids.
        test_idx: Test row ids.
        model_name: Regressor name.
        config: Parsed configuration.
        model_n_jobs: Threads for the model.

    Returns:
        Regression metric dict.
    """
    model = regressor_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train_idx], yr[train_idx])
    return regression_metrics(yr[test_idx], model.predict(X[test_idx]))


def run_stepwise(prepared: PreparedData, config: dict[str, Any],
                 splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, Any]:
    """Two-stage classify-then-regress-on-predicted-short (Tables 5-6), hybrid parallel.

    Phase 1 fits the RF/XGB prior classifiers serially (all cores) to get the
    G-mean threshold, the test-fold confusion (Table 5), and the predicted-short
    row ids per (prior, repeat). Phase 2 regresses on those rows (Table 6): SVM
    fits run in parallel across (prior, repeat); RF/XGB regressors run serially
    with all cores. Results are identical to the serial version.

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.
        splits: Optional precomputed splits.

    Returns:
        Dict with ``confusion`` and ``regression`` per prior.
    """
    X = prepared.X.to_numpy()
    yc = prepared.y_class.to_numpy()
    yr = prepared.y_reg.to_numpy()
    splits = splits or make_splits(len(X), config["evaluation"]["n_repeats"],
                                   config["evaluation"]["test_size"], config["project"]["random_state"])
    test_size = config["evaluation"]["test_size"]

    # Phase 1: prior classifiers (RF/XGB), serial with all cores.
    conf_by_prior: dict[str, list[dict]] = {prior: [] for prior in STEPWISE_PRIORS}
    short_idx: dict[tuple[str, int], tuple] = {}
    for prior in STEPWISE_PRIORS:
        for r, (train, test) in enumerate(splits):
            conf, tr_short, te_short = _prior_short_indices(X, yc, train, test, prior, config)
            conf_by_prior[prior].append(conf)
            short_idx[(prior, r)] = (tr_short, te_short)
            LOGGER.info("    stepwise prior %s: repeat %d/%d done", prior, r + 1, len(splits))
    LOGGER.info("stepwise phase 1 (priors) done")

    # Phase 2: predicted-short regression. SVM parallel; tree/linear serial (all cores).
    valid = [(p, r) for p in STEPWISE_PRIORS for r in range(len(splits)) if short_idx[(p, r)][0] is not None]
    per_reg: dict[str, dict[str, list]] = {p: {n: [None] * len(splits) for n in REGRESSOR_NAMES} for p in STEPWISE_PRIORS}

    svm_results = Parallel(n_jobs=_n_jobs(config), verbose=10)(
        delayed(_reg_on_idx)(X, yr, short_idx[(p, r)][0], short_idx[(p, r)][1], "SVM", config, 1) for p, r in valid)
    for (p, r), res in zip(valid, svm_results):
        per_reg[p]["SVM"][r] = res
    for name in [n for n in REGRESSOR_NAMES if n not in _PARALLEL_MODELS]:
        for i, (p, r) in enumerate(valid):
            per_reg[p][name][r] = _reg_on_idx(X, yr, short_idx[(p, r)][0], short_idx[(p, r)][1], name, config, -1)
            LOGGER.info("    stepwise regressor %s: %d/%d done", name, i + 1, len(valid))
    LOGGER.info("stepwise phase 2 (predicted-short regression) done")

    confusion: dict[str, Any] = {}
    regression: dict[str, pd.DataFrame] = {}
    for prior in STEPWISE_PRIORS:
        confusion[prior] = _summarise_confusion(conf_by_prior[prior], test_size)
        rows = {name: [m for m in per_reg[prior][name] if m is not None] for name in REGRESSOR_NAMES}
        regression[prior] = pd.DataFrame(
            {name: mean_metrics(vals) for name, vals in rows.items() if vals}
        ).T.reindex(REGRESSOR_NAMES)[REGRESSION_METRICS]

    return {"confusion": confusion, "regression": regression}


def _summarise_confusion(conf_rows: list[dict], test_size: float) -> dict[str, Any]:
    """Average per-repeat test confusion and estimate the full-cohort counts.

    Args:
        conf_rows: Per-repeat confusion dicts (test folds).
        test_size: Test fraction, used to scale test counts up to the full cohort.

    Returns:
        Dict with ``mean_test`` counts and ``full_cohort_estimate`` counts.
    """
    frame = pd.DataFrame(conf_rows)
    mean_test = frame.mean().to_dict()
    scale = 1.0 / test_size if test_size else 1.0
    full = {key: value * scale for key, value in mean_test.items()}
    return {"mean_test": mean_test, "full_cohort_estimate": full}
