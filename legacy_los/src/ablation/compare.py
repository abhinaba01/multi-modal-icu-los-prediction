"""Ablation: paired comparison of a baseline vs an augmented feature set.

This works entirely from the **per-repeat metrics the run scripts already saved**
(``results/<name>/tables/per_repeat/*.csv``) -- it does NOT refit any model. Both
feature sets were evaluated by their run scripts on the SAME repeated splits (the
splits are deterministic from the shared cohort size + seed), so per-repeat metric
differences are paired. A signature check guards that the two runs really scored
the identical set of stays before we compare them.

We report the mean delta (augmented - baseline) per model x metric and a paired
significance test (Wilcoxon signed-rank, paired t-test fallback) across repeats.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS

LOGGER = logging.getLogger(__name__)

_STAGE_METRICS = {
    "classification": CLASSIFICATION_METRICS,
    "regression_full": REGRESSION_METRICS,
    "regression_short": REGRESSION_METRICS,
}


def _paired_pvalue(baseline_vals: np.ndarray, augmented_vals: np.ndarray) -> float:
    """Paired significance of the metric difference across repeats.

    Args:
        baseline_vals: Per-repeat metric for the baseline.
        augmented_vals: Per-repeat metric for the augmented set.

    Returns:
        Two-sided p-value (Wilcoxon signed-rank; paired t-test fallback), or NaN
        when it cannot be computed (too few repeats or all-zero differences).
    """
    diffs = np.asarray(augmented_vals, dtype=float) - np.asarray(baseline_vals, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 3 or np.allclose(diffs, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        try:
            return float(stats.ttest_rel(augmented_vals, baseline_vals).pvalue)
        except Exception:  # noqa: BLE001
            return float("nan")


def _delta_and_pvalue(base_stage: dict[str, pd.DataFrame], aug_stage: dict[str, pd.DataFrame],
                      metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean deltas and paired p-values per model x metric for one stage.

    Args:
        base_stage: Baseline per-repeat metrics ``{model -> DataFrame}``.
        aug_stage: Augmented per-repeat metrics ``{model -> DataFrame}``.
        metrics: Metric column names.

    Returns:
        ``(delta_df, pvalue_df)`` indexed by model.
    """
    delta_rows, pvalue_rows = {}, {}
    for model in base_stage:
        if model not in aug_stage:
            continue
        base_df, aug_df = base_stage[model], aug_stage[model]
        delta_rows[model] = {m: float(aug_df[m].mean() - base_df[m].mean()) for m in metrics}
        pvalue_rows[model] = {m: _paired_pvalue(base_df[m].to_numpy(), aug_df[m].to_numpy()) for m in metrics}
    return pd.DataFrame(delta_rows).T[metrics], pd.DataFrame(pvalue_rows).T[metrics]


def _check_paired(base_meta: dict[str, Any], aug_meta: dict[str, Any], base_name: str, aug_name: str) -> None:
    """Guard that two runs scored the identical stays with the same repeat count.

    Args:
        base_meta: Baseline ``run_meta`` dict.
        aug_meta: Augmented ``run_meta`` dict.
        base_name: Baseline name (for messages).
        aug_name: Augmented name (for messages).

    Raises:
        ValueError: If the runs are not a paired comparison.
    """
    base_sig, aug_sig = base_meta.get("index_signature"), aug_meta.get("index_signature")
    if base_sig and aug_sig and base_sig != aug_sig:
        raise ValueError(
            f"Ablation requires paired runs, but '{base_name}' and '{aug_name}' scored different "
            f"cohorts (index signatures differ). Rebuild all feature tables together "
            f"(`build_features.py --which all`) and re-run the pipelines with the same config."
        )
    if base_meta.get("n_repeats") != aug_meta.get("n_repeats"):
        raise ValueError(
            f"'{base_name}' and '{aug_name}' used different n_repeats "
            f"({base_meta.get('n_repeats')} vs {aug_meta.get('n_repeats')}); re-run with the same config."
        )


def run_ablation_study(base_pr: dict[str, Any], aug_pr: dict[str, Any],
                       base_name: str = "baseline", aug_name: str = "augmented") -> dict[str, pd.DataFrame]:
    """Run one paired ablation study from saved per-repeat metrics (no refitting).

    Args:
        base_pr: Baseline per-repeat bundle from ``pipeline.load_per_repeat``.
        aug_pr: Augmented per-repeat bundle.
        base_name: Baseline name (for the pairing guard message).
        aug_name: Augmented name.

    Returns:
        Dict of result tables: per-stage deltas and paired p-values.
    """
    _check_paired(base_pr.get("meta", {}), aug_pr.get("meta", {}), base_name, aug_name)

    result: dict[str, pd.DataFrame] = {}
    key_map = {
        "classification": "classification",
        "regression_full": "regression",
        "regression_short": "regression_short",
    }
    for stage, metrics in _STAGE_METRICS.items():
        delta, pvalue = _delta_and_pvalue(base_pr[stage], aug_pr[stage], metrics)
        result[f"{key_map[stage]}_delta"] = delta
        result[f"{key_map[stage]}_pvalue"] = pvalue
    LOGGER.info("ablation %s vs %s computed from saved per-repeat metrics", base_name, aug_name)
    return result
