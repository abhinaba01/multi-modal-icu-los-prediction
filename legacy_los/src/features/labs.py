"""Laboratory features: mean of each of the 18 labs over the first 24h.

Faithful to the reference GitLab ``Extraction.sql``:

* **18 lab itemids** -- Phosphate (50970) is intentionally EXCLUDED even though
  the paper's Table 1 lists it, because the extraction SQL that produced the
  published results omits it (user-confirmed decision; see README).
* labevents has no ``stay_id``; join on ``hadm_id`` and window each stay by its
  own ICU ``intime`` (``(intime, intime+24h]``).
* mean of ``valuenum``; NO physiologic range filtering (the paper explicitly did
  not exclude any lab values).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.loader import MIMICLoader
from src.features.aggregate import chunked_windowed_mean

# labevents itemid -> output column (18 items; Phosphate 50970 deliberately absent).
LAB_ITEMIDS: dict[int, str] = {
    50868: "anion_gap",
    50882: "bicarbonate",
    50902: "chloride",
    50912: "creatinine",
    50931: "glucose",
    50960: "magnesium",
    50971: "potassium",
    50983: "sodium",
    51006: "urea_nitrogen",
    51221: "hematocrit",
    51222: "hemoglobin",
    51248: "mch",
    51249: "mchc",
    51250: "mcv",
    51265: "platelet_count",
    51277: "rdw",
    51279: "red_blood_cells",
    51301: "white_blood_cells",
}

LAB_COLUMNS = list(LAB_ITEMIDS.values())


def build_labs(loader: MIMICLoader, config: dict[str, Any], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 18 mean lab features for the cohort.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        cohort_df: Cohort with ``stay_id``, ``hadm_id`` and ``intime`` (datetime).

    Returns:
        DataFrame indexed by ``stay_id`` with the columns in :data:`LAB_COLUMNS`.
    """
    return chunked_windowed_mean(
        loader=loader,
        config=config,
        table="labevents",
        itemid_map=LAB_ITEMIDS,
        cohort_time_df=cohort_df[["hadm_id", "stay_id", "intime"]],
        join_key="hadm_id",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
    )
