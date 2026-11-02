"""B0: the internship two-stage XGBoost, trained on the FaTE split, plus TreeSHAP.

Stage 1 classifies long vs short stay; the G-mean threshold is chosen on train scores
(legacy ``gmean_threshold``). Stage 2 regresses LOS on the train stays predicted short.
Model factories, threshold rule and metrics are the legacy functions. Differences from
the internship, all required by CLAUDE.md A4 and recorded in docs/decisions.md:

* label encoding and min-max scaling are fit on train stays only (the internship fit
  them on the whole cohort; tree models are invariant to these monotone transforms);
* a category unseen in train becomes missing (XGBoost handles it natively).

SHAP values come from XGBoost's built-in TreeSHAP (``pred_contribs=True``), so no
``shap`` dependency is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from fate_icu.data.contract import Contract
from fate_icu.data.targets import long_stay_labels, round_los
from fate_icu.tabular._legacy import legacy


@dataclass
class TabularPreprocessor:
    """Legacy encoding (categorical codes in sorted order, then min-max), fit on train."""

    columns: list[str]
    categorical: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)
    mins: pd.Series | None = None
    spans: pd.Series | None = None

    def _encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        out = X[self.columns].copy()
        for column in self.categorical:
            values = out[column].astype(str)
            if fit:
                self.categories[column] = sorted(values.unique())
            codes = pd.Categorical(values, categories=self.categories[column]).codes.astype(float)
            codes[codes < 0] = np.nan
            out[column] = codes
        return out.apply(pd.to_numeric, errors="coerce")

    def fit(self, X: pd.DataFrame) -> TabularPreprocessor:
        encoded = self._encode(X, fit=True)
        self.mins = encoded.min()
        self.spans = (encoded.max() - self.mins).replace(0, np.nan)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mins is None:
            raise RuntimeError("fit the preprocessor first")
        scaled = (self._encode(X, fit=False) - self.mins) / self.spans
        constant = list(self.spans.index[self.spans.isna()])
        if constant:
            scaled[constant] = 0.0  # legacy maps a constant column to 0
        return scaled


@dataclass
class TwoStageXGB:
    """Fitted B0 model (see module docstring)."""

    feature_names: list[str]
    preprocessor: TabularPreprocessor
    classifier: Any
    regressor: Any | None
    gmean_threshold: float
    long_stay_threshold_days: float
    contract: Contract

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(X[self.feature_names]).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """``p_long``, hard labels at 0.5 and at the G-mean threshold, and LOS in days.

        ``los_days`` comes from the stage-2 regressor, which is trained on predicted-short
        stays only; for stays predicted long it is an extrapolation.
        """
        Xt = self.transform(X)
        p_long = self.classifier.predict_proba(Xt)[:, 1]
        out = pd.DataFrame(index=X.index)
        out["p_long"] = p_long
        out["pred_long_05"] = self.classifier.predict(Xt).astype(int)
        out["pred_long_gmean"] = (p_long >= self.gmean_threshold).astype(int)
        out["los_days"] = self.regressor.predict(Xt) if self.regressor is not None else np.nan
        return out

    def evaluate(self, X: pd.DataFrame, los: pd.Series) -> dict[str, Any]:
        """Legacy metrics on a held-out set, plus AUPRC and Brier score.

        Returns:
            ``classification`` (legacy accuracy/F1/balanced accuracy/MCC/AUROC at 0.5,
            AUPRC, Brier), ``stepwise_confusion`` at the G-mean threshold and
            ``stepwise_regression`` on predicted-short stays (legacy RMSE/MAE/MAPE/R² as
            squared Pearson correlation), with counts.
        """
        metrics_mod = legacy("evaluation.metrics")
        stepwise = legacy("models.stepwise")
        pred = self.predict(X)
        y = long_stay_labels(los, self.long_stay_threshold_days, self.contract).reindex(X.index).to_numpy()
        y_reg = round_los(los, self.contract).reindex(X.index).to_numpy()
        both_classes = len(np.unique(y)) > 1
        classification = metrics_mod.classification_metrics(
            y, pred["pred_long_05"].to_numpy(), pred["p_long"].to_numpy()
        )
        classification["auprc"] = (
            float(average_precision_score(y, pred["p_long"])) if both_classes else float("nan")
        )
        classification["brier"] = float(brier_score_loss(y, pred["p_long"], pos_label=1))
        short = (pred["pred_long_gmean"] == 0).to_numpy()
        regression = (
            metrics_mod.regression_metrics(y_reg[short], pred["los_days"].to_numpy()[short])
            if self.regressor is not None and short.sum() >= 2
            else {}
        )
        return {
            "n_stays": int(len(y)),
            "n_long": int(y.sum()),
            "classification": classification,
            "stepwise_confusion": stepwise.confusion_counts(y, pred["pred_long_gmean"].to_numpy()),
            "stepwise_regression": {"n_predicted_short": int(short.sum()), **regression},
        }


def train_two_stage_xgb(
    X: pd.DataFrame,
    los: pd.Series,
    split: pd.Series,
    *,
    threshold_days: float,
    contract: Contract,
    legacy_cfg: dict[str, Any],
) -> TwoStageXGB:
    """Fit B0 on the train split only.

    Args:
        X: Raw features (index ``stay_id``), e.g. ``load_static_features("fate_static")``.
        los: LOS in days (index ``stay_id``).
        split: Split label per stay; only ``train`` rows are used for fitting.
        threshold_days: Long-stay threshold from :func:`resolve_long_stay_threshold`.
        contract: The frozen contract.
        legacy_cfg: Legacy config (``legacy_config``) carrying seeds and model sizes.

    Returns:
        The fitted model.
    """
    classifiers = legacy("models.classifiers")
    regressors = legacy("models.regressors")
    stepwise = legacy("models.stepwise")
    n_jobs = int(legacy_cfg.get("compute", {}).get("n_jobs", -1))

    columns = list(X.columns)
    train = (split.reindex(X.index) == "train").to_numpy()
    if train.sum() < 2:
        raise ValueError("need at least two train stays")
    preprocessor = TabularPreprocessor(columns, contract.categorical(columns)).fit(X[train])
    Xt = preprocessor.transform(X).to_numpy(dtype=float)
    y_long = long_stay_labels(los, threshold_days, contract).reindex(X.index).to_numpy()
    y_reg = round_los(los, contract).reindex(X.index).to_numpy()
    if len(np.unique(y_long[train])) < 2:
        raise ValueError("the train split needs both short and long stays")

    classifier = classifiers.classifier_factories(legacy_cfg, n_jobs=n_jobs)[contract.b0.prior]()
    classifier.fit(Xt[train], y_long[train])
    train_scores = classifier.predict_proba(Xt[train])[:, 1]
    gmean = float(stepwise.gmean_threshold(y_long[train], train_scores))

    train_rows = np.flatnonzero(train)
    short_rows = train_rows[train_scores < gmean]
    regressor = None
    if len(short_rows) >= 2:
        regressor = regressors.regressor_factories(legacy_cfg, n_jobs=n_jobs)["XGBoost"]()
        regressor.fit(Xt[short_rows], y_reg[short_rows])
    return TwoStageXGB(columns, preprocessor, classifier, regressor, gmean, float(threshold_days), contract)


@dataclass
class ShapExplanation:
    """Per-stay TreeSHAP values on the classifier's log-odds (or the regressor's output)."""

    values: pd.DataFrame  # stays x features
    base_value: pd.Series  # per stay (constant for a tree ensemble)
    group_values: pd.DataFrame  # stays x contract feature groups (sums of values)


def explain_shap(model: TwoStageXGB, X: pd.DataFrame, stage: str = "classifier") -> ShapExplanation:
    """Exact TreeSHAP attributions for B0.

    Args:
        model: A fitted :class:`TwoStageXGB`.
        X: Raw features with ``model.feature_names``.
        stage: ``classifier`` (long-stay log-odds) or ``regressor`` (LOS days).

    Returns:
        Values whose row sums plus the base value equal the model's margin output.
    """
    import xgboost as xgb

    estimator = model.classifier if stage == "classifier" else model.regressor
    if estimator is None:
        raise ValueError(f"no fitted {stage}")
    booster = estimator.get_booster()
    contributions = booster.predict(xgb.DMatrix(model.transform(X)), pred_contribs=True)
    values = pd.DataFrame(contributions[:, :-1], index=X.index, columns=model.feature_names)
    base = pd.Series(contributions[:, -1], index=X.index, name="base_value")
    groups = pd.Series(model.contract.group_of(model.feature_names))
    group_values = values.T.groupby(groups).sum().T
    return ShapExplanation(values=values, base_value=base, group_values=group_values)
