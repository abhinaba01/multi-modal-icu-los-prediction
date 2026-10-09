"""Faithful preprocessing: the exact steps of the reference ``data_preparation.R``.

Order (mirrors the R script):

1. drop duplicate rows;
2. complete-case -- drop any row with a missing feature/target (R ``complete.cases``);
3. clean vital MEANS against the Table-2 physiologic ranges (the R code cleans the
   already-averaged per-stay values, not individual measurements);
4. label-encode the categorical features (``factor -> integer``, NOT one-hot);
5. min-max scale every feature (not LOS) to [0, 1] on the WHOLE dataset (the R
   code scales globally, before the repeated train/test splits);
6. build targets: regression ``round(los, 2)``; classification short(<=Q3)=0 /
   long(>Q3)=1, with the threshold = the 3rd quartile of LOS (paper's method;
   the R code hardcoded 4, which is ~Q3).

Encoding and scaling are fit on the whole dataset by design -- this reproduces the
reference pipeline exactly (the paper reports results from this procedure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_dataset import BASE_CATEGORICAL, TARGET

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Container for the model-ready dataset.

    Attributes:
        X: Scaled feature matrix (index = stay_id).
        y_reg: Regression target (LOS rounded to 2 dp).
        y_class: Classification target (0 = short, 1 = long).
        los: Raw (rounded) LOS in days, aligned to ``X`` (for actual-short subsets).
        threshold: LOS dichotomisation threshold in days (~Q3).
        feature_columns: Ordered list of feature column names.
    """

    X: pd.DataFrame
    y_reg: pd.Series
    y_class: pd.Series
    los: pd.Series
    threshold: float
    feature_columns: list[str]


def prepare_dataset(df: pd.DataFrame, config: dict[str, Any]) -> PreparedData:
    """Run the full faithful preprocessing on a raw feature table.

    Args:
        df: Feature table indexed by ``stay_id`` with ``los`` + feature columns.
        config: Parsed configuration.

    Returns:
        A :class:`PreparedData` bundle.

    Raises:
        ValueError: If the table is empty after complete-case + range cleaning.
    """
    frame = df.copy()
    feature_columns = [c for c in frame.columns if c != TARGET]

    # 1. drop duplicate rows
    before = len(frame)
    frame = frame.drop_duplicates()
    LOGGER.info("dropped %d duplicate rows", before - len(frame))

    # 2. complete-case: drop rows with any NA in features or target
    before = len(frame)
    frame = frame.dropna(subset=[TARGET, *feature_columns])
    LOGGER.info("complete-case: kept %d of %d rows", len(frame), before)

    # 3. clean vital means against Table-2 ranges
    frame = _clean_vital_means(frame, config)
    if frame.empty:
        raise ValueError("No rows remain after complete-case + range cleaning.")

    # 4. label-encode categoricals (factor -> integer)
    categorical = [c for c in feature_columns if c in BASE_CATEGORICAL]
    for column in categorical:
        frame[column] = pd.Categorical(frame[column].astype(str)).codes.astype(float)

    # 5. min-max scale every feature (not LOS)
    frame[feature_columns] = _min_max_scale(frame[feature_columns])

    # 6. targets
    los = frame[TARGET].astype(float).round(2)
    threshold = _resolve_threshold(los, config)
    y_class = (los > threshold).astype(int)
    LOGGER.info("classification threshold (Q3~) = %.3f days; long-stay prevalence = %.3f",
                threshold, y_class.mean())

    X = frame[feature_columns]
    return PreparedData(
        X=X,
        y_reg=los,
        y_class=y_class,
        los=los,
        threshold=threshold,
        feature_columns=feature_columns,
    )


def _clean_vital_means(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Drop stays whose mean vital falls outside its Table-2 range.

    Args:
        frame: Feature table (post complete-case).
        config: Parsed configuration (``vital_mean_ranges``).

    Returns:
        Filtered feature table.
    """
    ranges = config.get("vital_mean_ranges", {})
    out = frame
    for column, spec in ranges.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        low, high, inclusive = spec["low"], spec["high"], spec.get("inclusive", True)
        if inclusive:
            keep = (values >= low) & (values <= high)
        else:
            keep = (values > low) & (values < high)
        before = len(out)
        out = out[keep]
        LOGGER.info("range-clean %s [%s, %s] inclusive=%s: kept %d of %d",
                    column, low, high, inclusive, len(out), before)
    return out


def _min_max_scale(features: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale each column to [0, 1]; constant columns become 0.

    Args:
        features: Numeric feature matrix (categoricals already integer-encoded).

    Returns:
        Scaled feature matrix.
    """
    numeric = features.apply(pd.to_numeric, errors="coerce")
    col_min = numeric.min()
    col_max = numeric.max()
    span = (col_max - col_min).replace(0, np.nan)
    scaled = (numeric - col_min) / span
    return scaled.fillna(0.0)


def _resolve_threshold(los: pd.Series, config: dict[str, Any]) -> float:
    """Resolve the classification LOS threshold.

    Args:
        los: LOS values (days).
        config: Parsed configuration (``classification.threshold``).

    Returns:
        Threshold in days: the 3rd quartile when configured as ``"q3"``, else the
        numeric override.
    """
    setting = config["classification"]["threshold"]
    if isinstance(setting, str) and setting.lower() == "q3":
        return float(los.quantile(0.75))
    return float(setting)
