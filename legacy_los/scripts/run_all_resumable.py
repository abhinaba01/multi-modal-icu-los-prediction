"""Run all Hempel pipelines sequentially with resumable checkpoints.

This is a safer long-run entry point for machines that may sleep, hang, or be
rebooted mid-experiment. It runs the three feature sets in order:

    1. hempel
    2. hempel_sofa
    3. hempel_sofa_dynamics

Within each run, it checkpoints per-repeat metrics after each completed model in
classification/regression stages, and checkpoints stepwise prior/regressor work.
On restart, completed models are skipped and only missing work is recomputed.

Final outputs are written to the same ``results/<name>/tables`` layout as the
regular run scripts, including ``per_repeat/*.csv`` for ``run_ablation.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_fast_overrides, load_config  # noqa: E402
from src.evaluation.experiment import make_splits  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    classification_metrics,
    mean_metrics,
    regression_metrics,
)
from src.models.classifiers import CLASSIFIER_NAMES, classifier_factories  # noqa: E402
from src.models.regressors import REGRESSOR_NAMES, regressor_factories  # noqa: E402
from src.models.stepwise import confusion_counts, gmean_threshold  # noqa: E402
from src.pipeline import (  # noqa: E402
    feature_analysis,
    format_summary,
    load_feature_table,
    save_pipeline_results,
)
from src.preprocessing.prepare import PreparedData, prepare_dataset  # noqa: E402
from src.reporting import ensure_dir  # noqa: E402

LOGGER = logging.getLogger("run_all_resumable")

RUNS = [
    ("hempel", "hempel"),
    ("hempel_sofa", "hempel_sofa"),
    ("hempel_sofa_dynamics", "hempel_sofa_dynamics"),
]

STEPWISE_PRIORS = ["Random forest", "XGBoost"]
PARALLEL_REPEAT_MODELS = {"SVM"}
CONFUSION_COLUMNS = [
    "actual_long_pred_long",
    "actual_long_pred_short",
    "actual_short_pred_long",
    "actual_short_pred_short",
]


def configure_logging() -> None:
    """Configure console + file logging for the resumable runner."""
    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "run_all_resumable.log"), logging.StreamHandler()],
    )


def _n_jobs(config: dict[str, Any]) -> int:
    """Return the configured joblib worker count."""
    return int(config.get("compute", {}).get("n_jobs", -1))


def _slug(text: str) -> str:
    """Convert a display name into a stable file-name slug."""
    return text.lower().replace(" ", "_")


def _atomic_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a CSV via a temporary file, then atomically replace the target."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    """Write a compressed npz via a temporary file, then replace the target."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def _index_signature(index: pd.Index) -> str:
    """Stable hash of the scored stay-id index."""
    payload = ",".join(str(v) for v in index.tolist()).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _expected_final_files(out_dir: Path) -> list[Path]:
    """Files that indicate a run finished end-to-end."""
    return [
        out_dir / "table3_classification.csv",
        out_dir / "table4_regression.csv",
        out_dir / "table5_confusion_full_cohort_estimate.csv",
        out_dir / "table6_stepwise_regression_random_forest_prior.csv",
        out_dir / "table6_stepwise_regression_xgboost_prior.csv",
        out_dir / "fig4_correlations.csv",
        out_dir / "fig5_xgb_importance.csv",
        out_dir / "run_meta.csv",
        out_dir / "per_repeat" / "classification.csv",
        out_dir / "per_repeat" / "regression_full.csv",
        out_dir / "per_repeat" / "regression_short.csv",
    ]


def _final_complete(out_dir: Path) -> bool:
    """Return True when the canonical outputs for one run are all present."""
    return all(path.exists() for path in _expected_final_files(out_dir))


def _load_stage_checkpoint(out_dir: Path, checkpoint_dir: Path, stage: str,
                           expected_repeats: int, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """Load completed per-repeat model checkpoints for a stage."""
    candidates = [
        checkpoint_dir / f"{stage}.csv",
        out_dir / "per_repeat" / f"{stage}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}

    frame = pd.read_csv(path)
    needed = {"model", "repeat", *metrics}
    if not needed.issubset(frame.columns):
        LOGGER.warning("Ignoring incomplete checkpoint %s", path)
        return {}

    out: dict[str, pd.DataFrame] = {}
    for model, group in frame.groupby("model", sort=False):
        group = group.drop_duplicates("repeat", keep="last").sort_values("repeat")
        repeats = set(group["repeat"].astype(int))
        if set(range(expected_repeats)).issubset(repeats):
            out[model] = group.head(expected_repeats).drop(columns=["model", "repeat"]).reset_index(drop=True)
    return out


def _write_stage_checkpoint(checkpoint_dir: Path, stage: str, per_model: dict[str, pd.DataFrame]) -> None:
    """Persist per-repeat metrics for completed models in one stage."""
    frames = []
    for model, df in per_model.items():
        tidy = df.reset_index(drop=True).copy()
        tidy.insert(0, "model", model)
        tidy.insert(1, "repeat", range(len(tidy)))
        frames.append(tidy)
    if frames:
        _atomic_csv(pd.concat(frames, ignore_index=True), checkpoint_dir / f"{stage}.csv")


def _positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class scores for classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def _classification_repeat(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                           model_name: str, config: dict[str, Any], model_n_jobs: int) -> dict[str, float]:
    """Fit one classifier repeat and return metrics."""
    model = classifier_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train], y[train])
    return classification_metrics(y[test], model.predict(X[test]), _positive_scores(model, X[test]))


def _regression_repeat(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                       model_name: str, config: dict[str, Any], model_n_jobs: int) -> dict[str, float]:
    """Fit one regressor repeat and return metrics."""
    model = regressor_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train], y[train])
    return regression_metrics(y[test], model.predict(X[test]))


def _run_model_repeats(X: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
                       model_name: str, config: dict[str, Any],
                       repeat_fn: Callable[..., dict[str, float]],
                       label: str) -> pd.DataFrame:
    """Run all repeats for one model, parallelizing SVM across repeats."""
    if model_name in PARALLEL_REPEAT_MODELS:
        rows = Parallel(n_jobs=_n_jobs(config), verbose=10)(
            delayed(repeat_fn)(X, y, train, test, model_name, config, 1)
            for train, test in splits
        )
    else:
        rows = []
        for repeat, (train, test) in enumerate(splits, start=1):
            rows.append(repeat_fn(X, y, train, test, model_name, config, -1))
            LOGGER.info("%s | %s | repeat %d/%d done", label, model_name, repeat, len(splits))
    return pd.DataFrame(rows)


def _run_metric_stage(stage: str, label: str, X: np.ndarray, y: np.ndarray,
                      splits: list[tuple[np.ndarray, np.ndarray]], model_names: list[str],
                      metrics: list[str], config: dict[str, Any], checkpoint_dir: Path,
                      out_dir: Path, repeat_fn: Callable[..., dict[str, float]]) -> dict[str, pd.DataFrame]:
    """Run or resume a classification/regression stage checkpointed by model."""
    expected_repeats = len(splits)
    per_model = _load_stage_checkpoint(out_dir, checkpoint_dir, stage, expected_repeats, metrics)
    for model_name in model_names:
        if model_name in per_model:
            LOGGER.info("%s | %s already complete; skipping", label, model_name)
            continue
        LOGGER.info("%s | %s starting", label, model_name)
        per_model[model_name] = _run_model_repeats(X, y, splits, model_name, config, repeat_fn, label)
        _write_stage_checkpoint(checkpoint_dir, stage, per_model)
        LOGGER.info("%s | %s complete; checkpoint saved", label, model_name)
    return {name: per_model[name][metrics] for name in model_names}


def _mean_table(per_repeat: dict[str, pd.DataFrame], model_order: list[str], metrics: list[str]) -> pd.DataFrame:
    """Build the paper table from per-repeat metrics."""
    rows = {name: per_repeat[name].mean(numeric_only=True) for name in model_order}
    return pd.DataFrame(rows).T[metrics]


def _combine_regression(reg_full: pd.DataFrame, reg_short: pd.DataFrame) -> pd.DataFrame:
    """Combine full-range and actual-short regression metrics."""
    combined = reg_full.copy()
    for column in reg_short.columns:
        combined[f"{column}_short"] = reg_short[column]
    return combined


def _prior_path(checkpoint_dir: Path, prior: str, repeat: int) -> Path:
    """Path for one stepwise prior short-index checkpoint."""
    return checkpoint_dir / "stepwise_prior_short" / f"{_slug(prior)}_repeat_{repeat:02d}.npz"


def _load_short_indices(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a prior short-index checkpoint, returning None if it is unusable."""
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            return data["train_short"], data["test_short"]
    except Exception as exc:  # noqa: BLE001 - corrupt partial checkpoint; recompute
        LOGGER.warning("Could not read %s (%s); recomputing it", path, exc)
        return None


def _load_confusion_checkpoint(checkpoint_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load stepwise confusion checkpoints keyed by (prior, repeat)."""
    path = checkpoint_dir / "stepwise_confusion.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    needed = {"prior", "repeat", *CONFUSION_COLUMNS}
    if not needed.issubset(frame.columns):
        return {}
    out = {}
    for row in frame.to_dict("records"):
        out[(row["prior"], int(row["repeat"]))] = row
    return out


def _write_confusion_checkpoint(checkpoint_dir: Path, rows: dict[tuple[str, int], dict[str, Any]]) -> None:
    """Persist stepwise prior confusion checkpoints."""
    if rows:
        frame = pd.DataFrame(rows.values()).sort_values(["prior", "repeat"])
        _atomic_csv(frame, checkpoint_dir / "stepwise_confusion.csv")


def _fit_prior_short(X: np.ndarray, yc: np.ndarray, train: np.ndarray, test: np.ndarray,
                     prior: str, config: dict[str, Any]) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Fit one RF/XGB prior and return confusion plus predicted-short row ids."""
    clf = classifier_factories(config, n_jobs=-1)[prior]()
    clf.fit(X[train], yc[train])
    score_tr = _positive_scores(clf, X[train])
    score_te = _positive_scores(clf, X[test])
    threshold = gmean_threshold(yc[train], score_tr)
    train_short = train[score_tr < threshold]
    test_short = test[score_te < threshold]
    return confusion_counts(yc[test], score_te >= threshold), train_short, test_short


def _load_stepwise_regression(checkpoint_dir: Path) -> pd.DataFrame:
    """Load stepwise regression checkpoint rows."""
    path = checkpoint_dir / "stepwise_regression.csv"
    if not path.exists():
        return pd.DataFrame(columns=["prior", "model", "repeat", *REGRESSION_METRICS])
    frame = pd.read_csv(path)
    needed = {"prior", "model", "repeat", *REGRESSION_METRICS}
    if not needed.issubset(frame.columns):
        return pd.DataFrame(columns=["prior", "model", "repeat", *REGRESSION_METRICS])
    return frame.drop_duplicates(["prior", "model", "repeat"], keep="last")


def _write_stepwise_regression(checkpoint_dir: Path, frame: pd.DataFrame) -> None:
    """Persist stepwise regression checkpoint rows."""
    ordered = frame.sort_values(["prior", "model", "repeat"]).reset_index(drop=True)
    _atomic_csv(ordered, checkpoint_dir / "stepwise_regression.csv")


def _regression_on_indices(X: np.ndarray, yr: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                           model_name: str, config: dict[str, Any], model_n_jobs: int) -> dict[str, float]:
    """Fit one predicted-short regressor and score it."""
    model = regressor_factories(config, n_jobs=model_n_jobs)[model_name]()
    model.fit(X[train_idx], yr[train_idx])
    return regression_metrics(yr[test_idx], model.predict(X[test_idx]))


def _summarise_confusion(conf_rows: list[dict[str, Any]], test_size: float) -> dict[str, Any]:
    """Average test-fold confusion and estimate full-cohort counts."""
    frame = pd.DataFrame(conf_rows)[CONFUSION_COLUMNS]
    mean_test = frame.mean().to_dict()
    scale = 1.0 / test_size if test_size else 1.0
    return {
        "mean_test": mean_test,
        "full_cohort_estimate": {key: value * scale for key, value in mean_test.items()},
    }


def _run_stepwise(prepared: PreparedData, config: dict[str, Any],
                  splits: list[tuple[np.ndarray, np.ndarray]], checkpoint_dir: Path) -> dict[str, Any]:
    """Run or resume the stepwise RF/XGB-prior stage."""
    X = prepared.X.to_numpy()
    yc = prepared.y_class.to_numpy()
    yr = prepared.y_reg.to_numpy()
    test_size = float(config["evaluation"]["test_size"])
    conf_rows = _load_confusion_checkpoint(checkpoint_dir)
    short_idx: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for prior in STEPWISE_PRIORS:
        for repeat, (train, test) in enumerate(splits):
            key = (prior, repeat)
            stored = _load_short_indices(_prior_path(checkpoint_dir, prior, repeat))
            if key in conf_rows and stored is not None:
                short_idx[key] = stored
                LOGGER.info("stepwise prior | %s repeat %d/%d already complete; skipping",
                            prior, repeat + 1, len(splits))
                continue

            LOGGER.info("stepwise prior | %s repeat %d/%d starting", prior, repeat + 1, len(splits))
            conf, train_short, test_short = _fit_prior_short(X, yc, train, test, prior, config)
            _atomic_npz(_prior_path(checkpoint_dir, prior, repeat),
                        train_short=train_short, test_short=test_short)
            conf_rows[key] = {"prior": prior, "repeat": repeat, **conf}
            short_idx[key] = (train_short, test_short)
            _write_confusion_checkpoint(checkpoint_dir, conf_rows)
            LOGGER.info("stepwise prior | %s repeat %d/%d complete; checkpoint saved",
                        prior, repeat + 1, len(splits))

    reg_frame = _load_stepwise_regression(checkpoint_dir)
    all_new_rows = []
    for prior in STEPWISE_PRIORS:
        valid_repeats = [
            repeat for repeat in range(len(splits))
            if len(short_idx[(prior, repeat)][0]) >= 2 and len(short_idx[(prior, repeat)][1]) >= 2
        ]
        for model_name in REGRESSOR_NAMES:
            done = reg_frame[(reg_frame["prior"] == prior) & (reg_frame["model"] == model_name)]
            done_repeats = set(done["repeat"].astype(int)) if not done.empty else set()
            if set(valid_repeats).issubset(done_repeats):
                LOGGER.info("stepwise regression | %s prior | %s already complete; skipping", prior, model_name)
                continue

            LOGGER.info("stepwise regression | %s prior | %s starting", prior, model_name)
            if model_name in PARALLEL_REPEAT_MODELS:
                metrics = Parallel(n_jobs=_n_jobs(config), verbose=10)(
                    delayed(_regression_on_indices)(
                        X, yr, short_idx[(prior, repeat)][0], short_idx[(prior, repeat)][1],
                        model_name, config, 1,
                    )
                    for repeat in valid_repeats
                )
            else:
                metrics = []
                for i, repeat in enumerate(valid_repeats, start=1):
                    metrics.append(_regression_on_indices(
                        X, yr, short_idx[(prior, repeat)][0], short_idx[(prior, repeat)][1],
                        model_name, config, -1,
                    ))
                    LOGGER.info("stepwise regression | %s prior | %s repeat %d/%d done",
                                prior, model_name, i, len(valid_repeats))

            rows = [{"prior": prior, "model": model_name, "repeat": repeat, **metric}
                    for repeat, metric in zip(valid_repeats, metrics)]
            all_new_rows.extend(rows)
            reg_frame = pd.concat([reg_frame, pd.DataFrame(rows)], ignore_index=True)
            reg_frame = reg_frame.drop_duplicates(["prior", "model", "repeat"], keep="last")
            _write_stepwise_regression(checkpoint_dir, reg_frame)
            LOGGER.info("stepwise regression | %s prior | %s complete; checkpoint saved", prior, model_name)

    if all_new_rows:
        reg_frame = _load_stepwise_regression(checkpoint_dir)

    confusion = {}
    regression = {}
    for prior in STEPWISE_PRIORS:
        prior_conf = [conf_rows[(prior, repeat)] for repeat in range(len(splits))]
        confusion[prior] = _summarise_confusion(prior_conf, test_size)

        rows = {}
        for model_name in REGRESSOR_NAMES:
            group = reg_frame[(reg_frame["prior"] == prior) & (reg_frame["model"] == model_name)]
            rows[model_name] = mean_metrics(group[REGRESSION_METRICS].to_dict("records"))
        regression[prior] = pd.DataFrame(rows).T.reindex(REGRESSOR_NAMES)[REGRESSION_METRICS]

    return {"confusion": confusion, "regression": regression}


def _load_feature_analysis(checkpoint_dir: Path) -> tuple[pd.Series, pd.Series] | None:
    """Load feature-analysis checkpoints if both are present."""
    corr_path = checkpoint_dir / "fig4_correlations.csv"
    imp_path = checkpoint_dir / "fig5_xgb_importance.csv"
    if not corr_path.exists() or not imp_path.exists():
        return None
    correlations = pd.read_csv(corr_path, index_col=0).iloc[:, 0]
    correlations.name = "corr_with_los"
    importance = pd.read_csv(imp_path, index_col=0).iloc[:, 0]
    importance.name = "xgb_importance"
    return correlations, importance


def _run_feature_analysis(prepared: PreparedData, config: dict[str, Any],
                          checkpoint_dir: Path) -> tuple[pd.Series, pd.Series]:
    """Run or resume feature analysis."""
    cached = _load_feature_analysis(checkpoint_dir)
    if cached is not None:
        LOGGER.info("feature analysis already complete; skipping")
        return cached
    LOGGER.info("feature analysis starting")
    correlations, importance = feature_analysis(prepared, config)
    _atomic_csv(correlations.to_frame(), checkpoint_dir / "fig4_correlations.csv", index=True)
    _atomic_csv(importance.to_frame(), checkpoint_dir / "fig5_xgb_importance.csv", index=True)
    LOGGER.info("feature analysis complete; checkpoint saved")
    return correlations, importance


def _run_one(config: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    """Run or resume one feature-set pipeline."""
    out_dir = ensure_dir(PROJECT_ROOT / config["output"]["results_dir"] / name / "tables")
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")

    LOGGER.info("=== %s: loading and preparing feature table ===", name)
    prepared = prepare_dataset(load_feature_table(config, key), config)
    n_repeats = int(config["evaluation"]["n_repeats"])
    test_size = float(config["evaluation"]["test_size"])
    seed = int(config["project"]["random_state"])
    splits = make_splits(len(prepared.X), n_repeats, test_size, seed)
    short_mask = (prepared.los <= prepared.threshold).to_numpy()
    short_splits = make_splits(int(short_mask.sum()), n_repeats, test_size, seed)

    LOGGER.info("=== %s: Table 3 classification ===", name)
    clf_pr = _run_metric_stage(
        "classification", f"{name} classification", prepared.X.to_numpy(), prepared.y_class.to_numpy(),
        splits, CLASSIFIER_NAMES, CLASSIFICATION_METRICS, config, checkpoint_dir, out_dir,
        _classification_repeat,
    )

    LOGGER.info("=== %s: Table 4 regression full ===", name)
    reg_full_pr = _run_metric_stage(
        "regression_full", f"{name} regression_full", prepared.X.to_numpy(), prepared.y_reg.to_numpy(),
        splits, REGRESSOR_NAMES, REGRESSION_METRICS, config, checkpoint_dir, out_dir,
        _regression_repeat,
    )

    LOGGER.info("=== %s: Table 4 regression actual-short ===", name)
    reg_short_pr = _run_metric_stage(
        "regression_short", f"{name} regression_short",
        prepared.X.iloc[short_mask].to_numpy(), prepared.y_reg.iloc[short_mask].to_numpy(),
        short_splits, REGRESSOR_NAMES, REGRESSION_METRICS, config, checkpoint_dir, out_dir,
        _regression_repeat,
    )

    LOGGER.info("=== %s: Tables 5-6 stepwise ===", name)
    stepwise = _run_stepwise(prepared, config, splits, checkpoint_dir)

    LOGGER.info("=== %s: feature analysis ===", name)
    correlations, importance = _run_feature_analysis(prepared, config, checkpoint_dir)

    table3 = _mean_table(clf_pr, CLASSIFIER_NAMES, CLASSIFICATION_METRICS)
    table4 = _combine_regression(
        _mean_table(reg_full_pr, REGRESSOR_NAMES, REGRESSION_METRICS),
        _mean_table(reg_short_pr, REGRESSOR_NAMES, REGRESSION_METRICS),
    )

    results = {
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
            "index_signature": _index_signature(prepared.X.index),
        },
    }
    results["out_dir"] = save_pipeline_results(results, name, config)
    return results


def main() -> None:
    """Entry point for the sequential resumable runner."""
    parser = argparse.ArgumentParser(description="Run all Hempel pipelines sequentially with resume checkpoints.")
    parser.add_argument("--fast", action="store_true", help="Use smoke-test model sizes/repeats.")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=[name for _, name in RUNS],
        default=[name for _, name in RUNS],
        help="Subset of runs to execute, in the standard order.",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config()
    if args.fast:
        config = apply_fast_overrides(config)
        LOGGER.warning("--fast is active; outputs are smoke-test results, not paper-faithful results")

    selected = set(args.runs)
    for key, name in RUNS:
        if name not in selected:
            continue
        out_dir = ensure_dir(PROJECT_ROOT / config["output"]["results_dir"] / name / "tables")
        if _final_complete(out_dir):
            LOGGER.info("=== %s already has final outputs; skipping ===", name)
            continue
        results = _run_one(config, key, name)
        print("\n" + format_summary(results))
        print(f"\n{name} results written to: {results['out_dir']}")

    LOGGER.info("All requested runs are complete.")


if __name__ == "__main__":
    main()
