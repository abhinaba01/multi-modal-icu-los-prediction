"""End-to-end pipeline shared by the three run scripts.

Loads a prepared dataset, runs the paper's four experiments (Tables 3-6) plus the
lightweight feature analysis (Fig 4 correlations, Fig 5 XGBoost importance), and
writes every table to ``results/<name>/``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, MissingDataError, processed_path
from src.evaluation.experiment import (classification_per_repeat, make_splits, mean_table,
                                       regression_per_repeat, run_stepwise)
from src.evaluation.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from src.models.classifiers import CLASSIFIER_NAMES, classifier_factories
from src.models.regressors import REGRESSOR_NAMES
from src.reporting import ensure_dir, save_table
from src.preprocessing.prepare import PreparedData, prepare_dataset

# Per-repeat metric stages saved for the ablation to reuse (no refitting).
PER_REPEAT_STAGES = ["classification", "regression_full", "regression_short"]

# Approximate share of a full run's wall-time per stage (from benchmarking), used
# only to log a rough "time remaining" estimate -- not exact.
_STAGE_WEIGHTS = {
    "classification": 0.15,
    "regression_full": 0.24,
    "regression_short": 0.17,
    "stepwise": 0.43,
    "feature_analysis": 0.01,
}


def _fmt_duration(seconds: float) -> str:
    """Format a duration as e.g. ``1h23m`` / ``45m`` / ``30s``.

    Args:
        seconds: Duration in seconds.

    Returns:
        Compact human-readable string.
    """
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _Progress:
    """Logs elapsed time and a rough ETA after each pipeline stage."""

    def __init__(self) -> None:
        self.start = time.time()
        self.done_weight = 0.0

    def stage_done(self, name: str) -> None:
        """Record a finished stage and log elapsed + estimated remaining time.

        Args:
            name: Stage name (key of :data:`_STAGE_WEIGHTS`).
        """
        self.done_weight += _STAGE_WEIGHTS.get(name, 0.0)
        elapsed = time.time() - self.start
        frac = min(max(self.done_weight, 1e-6), 1.0)
        remaining = elapsed * (1.0 - frac) / frac
        LOGGER.info("[progress] %-18s done | elapsed %s | ~%s remaining (%.0f%% of run)",
                    name, _fmt_duration(elapsed), _fmt_duration(remaining), frac * 100)

LOGGER = logging.getLogger(__name__)


def load_feature_table(config: dict[str, Any], key: str) -> pd.DataFrame:
    """Load a processed feature parquet, guarding when it has not been built.

    Args:
        config: Parsed configuration.
        key: One of ``hempel``, ``hempel_sofa``, ``hempel_sofa_dynamics``.

    Returns:
        Feature table indexed by ``stay_id``.

    Raises:
        MissingDataError: If the parquet is absent (with a build hint).
    """
    path = processed_path(config, key)
    if not Path(path).exists():
        raise MissingDataError(
            f"Feature table '{key}' not found at {path}.\n"
            f"Build it first:  python scripts/build_features.py --which {_which_flag(key)}"
        )
    df = pd.read_parquet(path)
    if df.index.name != "stay_id":
        if "stay_id" in df.columns:
            df = df.set_index("stay_id")
    return df


def _which_flag(key: str) -> str:
    """Map a feature key to the ``build_features.py --which`` flag.

    Args:
        key: Feature key.

    Returns:
        The corresponding ``--which`` value.
    """
    return {"hempel": "hempel", "hempel_sofa": "sofa", "hempel_sofa_dynamics": "dynamics"}[key]


def run_full_pipeline(prepared: PreparedData, config: dict[str, Any]) -> dict[str, Any]:
    """Run all four experiments plus feature analysis.

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.

    Returns:
        Dict with ``table3``, ``table4``, ``stepwise``, ``correlations``,
        ``xgb_importance`` and ``meta``.
    """
    seed = config["project"]["random_state"]
    n_repeats = config["evaluation"]["n_repeats"]
    test_size = config["evaluation"]["test_size"]
    splits = make_splits(len(prepared.X), n_repeats, test_size, seed)
    n_short = int((prepared.los <= prepared.threshold).sum())
    short_splits = make_splits(n_short, n_repeats, test_size, seed)
    progress = _Progress()

    # Compute per-repeat metrics ONCE; the mean tables (3/4) are derived from them,
    # and the per-repeat frames are saved so the ablation can reuse them without
    # refitting. Using the shared splits keeps the ablation a paired comparison.
    LOGGER.info("=== Table 3: classification (per-repeat) ===")
    clf_pr = classification_per_repeat(prepared, config, splits)
    table3 = mean_table(clf_pr, CLASSIFIER_NAMES, CLASSIFICATION_METRICS)
    progress.stage_done("classification")

    LOGGER.info("=== Table 4: regression full + actual-short (per-repeat) ===")
    reg_full_pr = regression_per_repeat(prepared, config, splits, actual_short_only=False)
    progress.stage_done("regression_full")
    reg_short_pr = regression_per_repeat(prepared, config, short_splits, actual_short_only=True)
    progress.stage_done("regression_short")
    table4 = _combine_regression(mean_table(reg_full_pr, REGRESSOR_NAMES, REGRESSION_METRICS),
                                 mean_table(reg_short_pr, REGRESSOR_NAMES, REGRESSION_METRICS))

    LOGGER.info("=== Tables 5-6: stepwise ===")
    stepwise = run_stepwise(prepared, config, splits)
    progress.stage_done("stepwise")

    LOGGER.info("=== Feature analysis (Fig 4/5) ===")
    correlations, importance = feature_analysis(prepared, config)
    progress.stage_done("feature_analysis")

    return {
        "table3": table3,
        "table4": table4,
        "stepwise": stepwise,
        "correlations": correlations,
        "xgb_importance": importance,
        "per_repeat": {
            "classification": clf_pr,
            "regression_full": reg_full_pr,
            "regression_short": reg_short_pr,
        },
        "meta": {
            "n_stays": int(len(prepared.X)),
            "n_features": int(prepared.X.shape[1]),
            "threshold_days": float(prepared.threshold),
            "long_prevalence": float(prepared.y_class.mean()),
            "n_repeats": int(n_repeats),
            # Signature of the exact stays (and order) this run scored, so the
            # ablation can assert two feature sets are paired before comparing.
            "index_signature": _index_signature(prepared.X.index),
        },
    }


def _index_signature(index: pd.Index) -> str:
    """Stable hash of an ordered stay-id index (for paired-ablation checks).

    Args:
        index: The ``stay_id`` index of a prepared feature matrix.

    Returns:
        A short hex digest of the ordered ids.
    """
    payload = ",".join(str(v) for v in index.tolist()).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _combine_regression(reg_full: pd.DataFrame, reg_short: pd.DataFrame) -> pd.DataFrame:
    """Combine full-range and actual-short regression into one Table-4 view.

    Args:
        reg_full: Metrics on the full LOS range.
        reg_short: Metrics on actual short stays (parenthetical column).

    Returns:
        DataFrame with ``<metric>`` and ``<metric>_short`` columns.
    """
    combined = reg_full.copy()
    for col in reg_short.columns:
        combined[f"{col}_short"] = reg_short[col]
    return combined


def feature_analysis(prepared: PreparedData, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Feature/LOS correlations (Fig 4) and XGBoost importance (Fig 5).

    Args:
        prepared: Prepared dataset.
        config: Parsed configuration.

    Returns:
        ``(correlations, importance)`` -- correlation of each feature with LOS
        (sorted by magnitude) and XGBoost classification feature importances.
    """
    correlations = prepared.X.corrwith(prepared.y_reg).sort_values(key=lambda s: s.abs(), ascending=False)
    correlations.name = "corr_with_los"

    xgb = classifier_factories(config)["XGBoost"]()
    xgb.fit(prepared.X, prepared.y_class)
    importance = pd.Series(xgb.feature_importances_, index=prepared.X.columns, name="xgb_importance")
    importance = importance.sort_values(ascending=False)
    return correlations, importance


def save_pipeline_results(results: dict[str, Any], name: str, config: dict[str, Any]) -> Path:
    """Write all pipeline tables under ``results/<name>/tables``.

    Args:
        results: Output of :func:`run_full_pipeline`.
        name: Result namespace (``hempel``, ``hempel_sofa``, ...).
        config: Parsed configuration.

    Returns:
        The output directory.
    """
    out_dir = ensure_dir(PROJECT_ROOT / config["output"]["results_dir"] / name / "tables")

    save_table(results["table3"], out_dir, "table3_classification")
    save_table(results["table4"], out_dir, "table4_regression")

    for prior, df in results["stepwise"]["regression"].items():
        save_table(df, out_dir, f"table6_stepwise_regression_{_slug(prior)}_prior")

    confusion = pd.DataFrame({prior: results["stepwise"]["confusion"][prior]["full_cohort_estimate"]
                              for prior in results["stepwise"]["confusion"]})
    save_table(confusion, out_dir, "table5_confusion_full_cohort_estimate", float_format="%.0f")

    save_table(results["correlations"].to_frame(), out_dir, "fig4_correlations")
    save_table(results["xgb_importance"].to_frame(), out_dir, "fig5_xgb_importance")

    meta = pd.DataFrame([results["meta"]])
    save_table(meta, out_dir, "run_meta")

    if "per_repeat" in results:
        _save_per_repeat(results["per_repeat"], out_dir / "per_repeat")
    return out_dir


def _save_per_repeat(per_repeat: dict[str, dict[str, pd.DataFrame]], out_dir: Path) -> None:
    """Write per-repeat metrics as tidy CSVs (one per stage) for ablation reuse.

    Args:
        per_repeat: ``stage -> {model -> DataFrame(rows=repeat, cols=metrics)}``.
        out_dir: Destination directory.
    """
    directory = ensure_dir(out_dir)
    for stage, models in per_repeat.items():
        frames = []
        for model, df in models.items():
            tidy = df.reset_index(drop=True).copy()
            tidy.insert(0, "model", model)
            tidy.insert(1, "repeat", range(len(tidy)))
            frames.append(tidy)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(directory / f"{stage}.csv", index=False)
    LOGGER.info("wrote per-repeat metrics to %s", directory)


def load_per_repeat(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Load a run's saved per-repeat metrics (for the ablation to reuse).

    Args:
        config: Parsed configuration.
        name: Result namespace (``hempel``, ``hempel_sofa``, ...).

    Returns:
        Dict with each stage mapped to ``{model -> DataFrame(rows=repeat)}`` plus a
        ``meta`` entry (from ``run_meta.csv``, including ``index_signature``).

    Raises:
        MissingDataError: If the per-repeat artifacts are absent (run the script first).
    """
    tables_dir = PROJECT_ROOT / config["output"]["results_dir"] / name / "tables"
    pr_dir = tables_dir / "per_repeat"
    if not pr_dir.exists():
        raise MissingDataError(
            f"Per-repeat metrics for '{name}' not found at {pr_dir}.\n"
            f"Run the pipeline first, e.g.:  python scripts/run_{name}.py"
        )
    out: dict[str, Any] = {}
    for stage in PER_REPEAT_STAGES:
        frame = pd.read_csv(pr_dir / f"{stage}.csv")
        out[stage] = {
            model: group.sort_values("repeat").drop(columns=["model", "repeat"]).reset_index(drop=True)
            for model, group in frame.groupby("model", sort=False)
        }
    meta_path = tables_dir / "run_meta.csv"
    out["meta"] = pd.read_csv(meta_path).iloc[0].to_dict() if meta_path.exists() else {}
    return out


def _slug(text: str) -> str:
    """Lower-case, underscore-separated slug for file names.

    Args:
        text: Input text.

    Returns:
        Slugified text.
    """
    return text.lower().replace(" ", "_")


def prepare_from_key(config: dict[str, Any], key: str) -> PreparedData:
    """Load a feature table by key and run the faithful preprocessing.

    Args:
        config: Parsed configuration.
        key: Feature key.

    Returns:
        Prepared dataset.
    """
    df = load_feature_table(config, key)
    return prepare_dataset(df, config)


def run_and_report(config: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    """Prepare, run the full pipeline, save results, and return them.

    Args:
        config: Parsed configuration.
        key: Feature key to load.
        name: Result namespace for outputs.

    Returns:
        The pipeline results dict (with an extra ``out_dir`` key).
    """
    prepared = prepare_from_key(config, key)
    results = run_full_pipeline(prepared, config)
    out_dir = save_pipeline_results(results, name, config)
    results["out_dir"] = out_dir
    return results


def format_summary(results: dict[str, Any]) -> str:
    """Render a compact console summary of the main tables.

    Args:
        results: Output of :func:`run_full_pipeline`.

    Returns:
        A human-readable multi-line string.
    """
    meta = results["meta"]
    lines = [
        f"Cohort: {meta['n_stays']} stays | {meta['n_features']} features | "
        f"threshold={meta['threshold_days']:.2f}d | long-prevalence={meta['long_prevalence']:.3f} | "
        f"repeats={meta['n_repeats']}",
        "",
        "Table 3 - Classification (mean over repeats):",
        results["table3"].round(3).to_string(),
        "",
        "Table 4 - Regression [full | _short = actual short]:",
        results["table4"].round(3).to_string(),
        "",
        "Table 6 - Stepwise regression on predicted-short:",
    ]
    for prior, df in results["stepwise"]["regression"].items():
        lines.append(f"  prior = {prior}:")
        lines.append(df.round(3).to_string())
    return "\n".join(lines)
