"""Vital-sign features: mean of each of the 7 vitals over the first 24h.

Faithful to the reference: chartevents itemids only, window ``(intime, intime+24h]``,
mean of ``valuenum``. Temperature is kept in DEGREES FAHRENHEIT (itemid 223761) --
the R code cleans it against the [86, 113] F range and never converts to Celsius;
min-max scaling later removes the unit dependence. No per-measurement outlier
filtering here -- the paper's range cleaning (Table 2) is applied to the per-stay
MEAN in ``preprocessing/prepare.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.loader import MIMICLoader
from src.features.aggregate import chunked_windowed_mean

# chartevents itemid -> output column. Order matches the reference vital list.
VITAL_ITEMIDS: dict[int, str] = {
    220045: "heart_rate",
    220210: "respiration",
    220277: "saturation",
    223761: "temperature",   # degrees Fahrenheit (kept as-is)
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

VITAL_COLUMNS = list(VITAL_ITEMIDS.values())


def build_vitals(loader: MIMICLoader, config: dict[str, Any], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 7 mean vital-sign features for the cohort.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        cohort_df: Cohort with ``stay_id`` and ``intime`` (datetime).

    Returns:
        DataFrame indexed by ``stay_id`` with the columns in :data:`VITAL_COLUMNS`.
    """
    return chunked_windowed_mean(
        loader=loader,
        config=config,
        table="chartevents",
        itemid_map=VITAL_ITEMIDS,
        cohort_time_df=cohort_df[["stay_id", "intime"]],
        join_key="stay_id",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
    )
