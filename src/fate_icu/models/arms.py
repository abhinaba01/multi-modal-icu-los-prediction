"""The comparison arms and their evaluation (Stage 3).

====  =========================================================  =============
Arm   What it is                                                 Trained on
====  =========================================================  =============
B0    internship two-stage XGBoost (``fate_icu.tabular``)         all stays
B1    static MLP                                                  all stays
B2    time series only (LSTM)                                     all stays
B3    static + time series, concatenation + MLP                   all stays
B4    image only (frozen DenseNet features + MLP)                 paired stays
B5    late fusion: mean of the B1, B2 and B4 logits                (B1, B2, B4)
B6    static + time series + image, concatenation + MLP           all stays
M     attention fusion (ours), missing modalities masked          all stays
====  =========================================================  =============

Populations: P1 = test stays with an eligible X-ray (primary), P2 = all test stays. B4 has no
prediction without an image, so it is reported on P1 only. Every arm is scored on the same
test patients (CLAUDE.md A4); thresholds are fitted on train scores only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fate_icu.eval.predictive import evaluate_population
from fate_icu.models.data import StayData
from fate_icu.models.fusion import ModelSpec
from fate_icu.models.train import TrainConfig, predict, train_model
from fate_icu.tabular._legacy import legacy

POPULATIONS = ("P1", "P2")


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    spec: ModelSpec | None = None  # None for late fusion
    paired_only: bool = False  # train and select on stays that have an image
    populations: tuple[str, ...] = POPULATIONS


ARMS: dict[str, Arm] = {
    "B1": Arm("B1", "static MLP", ModelSpec(("static",), "none")),
    "B2": Arm("B2", "time series (LSTM)", ModelSpec(("ts",), "none")),
    "B3": Arm("B3", "static + time series, concat", ModelSpec(("static", "ts"), "concat")),
    "B4": Arm("B4", "image only", ModelSpec(("image",), "none"), paired_only=True, populations=("P1",)),
    "B5": Arm("B5", "late fusion (mean logit of B1, B2, B4)", None),
    "B6": Arm("B6", "static + time series + image, concat", ModelSpec(("static", "ts", "image"), "concat")),
    "M": Arm("M", "attention fusion", ModelSpec(("static", "ts", "image"), "attention")),
    "M_transformer": Arm(
        "M_transformer",
        "attention fusion, Transformer time series (ablation)",
        ModelSpec(("static", "ts", "image"), "attention", ts_encoder="transformer"),
    ),
}
LATE_FUSION_PARTS = ("B1", "B2", "B4")


def sized(arm: Arm, d: int, heads: int, layers: int, dropout: float) -> Arm:
    """The arm with a different model width (the smoke config uses a small network)."""
    if arm.spec is None:
        return arm
    return replace(arm, spec=replace(arm.spec, d=d, heads=heads, layers=layers, dropout=dropout))


def late_fusion(parts: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Mean logit (and mean log1p LOS) of the unimodal predictions that exist for each stay."""
    logit = np.stack([p["logit"] for p in parts.values()])
    log_los = np.log1p(np.stack([p["los_days"] for p in parts.values()]))
    n = np.sum(~np.isnan(logit), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_logit = np.where(n > 0, np.nansum(logit, axis=0) / np.maximum(n, 1), np.nan)
        mean_log_los = np.where(n > 0, np.nansum(log_los, axis=0) / np.maximum(n, 1), np.nan)
    return {"logit": mean_logit, "p_long": 1 / (1 + np.exp(-mean_logit)), "los_days": np.expm1(mean_log_los)}


def predict_all(trained, data: StayData) -> dict[str, np.ndarray]:
    """Predictions for every stay (train, val and test) in ``data`` order."""
    return predict(trained, data, np.arange(len(data)))


def run_arm(
    arm: Arm, data: StayData, cfg: TrainConfig, seed: int, keep: dict | None = None
) -> dict[str, np.ndarray]:
    """Train one neural arm on one seed; predictions for every stay.

    ``keep``, when given, receives the trained model under the arm name.
    """
    if arm.spec is None:
        raise ValueError(f"{arm.name} is not trained; build it with late_fusion")
    train_rows = data.rows("train", paired_only=arm.paired_only)
    val_rows = data.rows("val", paired_only=arm.paired_only)
    trained = train_model(arm.spec, data, cfg, seed, train_rows=train_rows, val_rows=val_rows)
    if keep is not None:
        keep[arm.name] = trained
    return predict_all(trained, data)


def run_arms(
    names: list[str],
    data: StayData,
    cfg: TrainConfig,
    seed: int,
    arms: dict[str, Arm] | None = None,
    keep: dict | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Predictions of the requested neural arms and B5 for one seed (B5 needs B1, B2 and B4).

    ``keep``, when given, receives every trained network (B5 has none: it is a mean).
    """
    arms = arms or ARMS
    wanted = set(names) | (set(LATE_FUSION_PARTS) if "B5" in names else set())
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name in arms:
        if name in wanted and arms[name].spec is not None:
            predictions[name] = run_arm(arms[name], data, cfg, seed, keep)
    if "B5" in names:
        predictions["B5"] = late_fusion({k: predictions[k] for k in LATE_FUSION_PARTS})
    return {name: predictions[name] for name in names}


def evaluate_arm(
    pred: dict[str, np.ndarray],
    data: StayData,
    population: str,
    *,
    n_boot: int = 1000,
    seed: int = 666,
) -> dict:
    """Metrics with patient-level CIs on the test stays of a population, or a skip reason.

    The two-stage regression of the internship is also reported: MAE and RMSE over the stays
    predicted short at the G-mean threshold of the train scores.
    """
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}")
    test = data.rows("test", paired_only=population == "P1")
    test = test[~np.isnan(pred["p_long"][test])]
    if len(test) < 2 or len(np.unique(data.y_long[test])) < 2:
        return {"skipped": "fewer than two test stays or only one class", "n_stays": int(len(test))}
    train = data.rows("train")
    train = train[~np.isnan(pred["p_long"][train])]
    threshold = float(legacy("models.stepwise").gmean_threshold(data.y_long[train], pred["p_long"][train]))
    result = evaluate_population(
        data.y_long[test],
        pred["p_long"][test],
        data.los_days[test],
        pred["los_days"][test],
        data.subject_ids[test],
        n_boot=n_boot,
        seed=seed,
    )
    short = test[pred["p_long"][test] < threshold]
    result["n_stays"] = int(len(test))
    result["n_patients"] = int(len(np.unique(data.subject_ids[test])))
    result["gmean_threshold"] = threshold
    result["n_predicted_short"] = int(len(short))
    if len(short) >= 2:
        regression = evaluate_population(
            data.y_long[short],
            pred["p_long"][short],
            data.los_days[short],
            pred["los_days"][short],
            data.subject_ids[short],
            n_boot=n_boot,
            seed=seed,
            classification={},
        )
        result["pred_short"] = {k: v for k, v in regression.items() if k.endswith("_days")}
    return result
