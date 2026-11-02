"""Choose each ICU stay's chest X-ray: the patient's own most recent frontal image before t0.

Rules (Stage 1; docs/decisions.md D-025):

* frontal views only: ``ViewPosition`` exactly ``AP`` or ``PA`` (spaces and case ignored);
* images are matched on ``subject_id`` (MIMIC-CXR has no ``hadm_id`` or ``stay_id``) and
  are eligible when ``intime - lookback <= study_datetime <= t0``, both ends included;
* the most recent eligible image wins; ties go to AP over PA, then the smallest
  ``dicom_id``, so the result never depends on row order;
* every cohort stay is kept, with ``has_cxr``; an image may serve two stays of the same
  patient when their windows overlap.

The choice uses only the patient, the times and the view, never LOS or any label
(CLAUDE.md A4, real pairings only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fate_icu.data.cxr.metadata import parse_study_datetimes

FRONTAL_VIEWS = ("AP", "PA")
VIEW_RANK = {"AP": 0, "PA": 1}
PAIRED_COLUMNS = [
    "stay_id",
    "subject_id",
    "study_id",
    "dicom_id",
    "study_datetime",
    "hours_before_t0",
    "view",
    "has_cxr",
]


@dataclass
class Linkage:
    """The pairing for one cohort.

    Attributes:
        paired: One row per cohort stay, columns :data:`PAIRED_COLUMNS`.
        flow: Counts after each step.
        candidates: Every eligible (stay, image) pair, before choosing one per stay.
    """

    paired: pd.DataFrame
    flow: dict[str, int] = field(default_factory=dict)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


def frontal_view(view_position: pd.Series) -> pd.Series:
    """``AP`` / ``PA`` for frontal views, NaN otherwise (lateral, other or missing)."""
    cleaned = view_position.astype(str).str.strip().str.upper()
    return cleaned.where(view_position.notna() & cleaned.isin(FRONTAL_VIEWS), np.nan)


def link(metadata: pd.DataFrame, cohort: pd.DataFrame, hours: int, lookback_hours: float = 24.0) -> Linkage:
    """Pair each cohort stay with its most recent eligible frontal image.

    Args:
        metadata: MIMIC-CXR metadata (``read_metadata`` or the synthetic generator).
        cohort: Index ``stay_id``; columns ``subject_id`` and ``intime``.
        hours: The window W (t0 = intime + W).
        lookback_hours: How long before ICU admission an image may be.

    Returns:
        The linkage, with every stay in ``paired``.
    """
    if lookback_hours < 0:
        raise ValueError("lookback_hours must be >= 0")
    meta = metadata.drop_duplicates(subset="dicom_id").copy()
    flow = {"metadata_images": int(len(metadata)), "unique_images": int(len(meta))}
    meta["subject_id"] = pd.to_numeric(meta["subject_id"]).astype("int64")
    meta["study_id"] = pd.to_numeric(meta["study_id"]).astype("int64")
    meta["study_datetime"] = parse_study_datetimes(meta["StudyDate"], meta["StudyTime"]).to_numpy()
    meta = meta[meta["study_datetime"].notna()]
    flow["with_valid_study_datetime"] = int(len(meta))
    meta["view"] = frontal_view(meta["ViewPosition"]).to_numpy()
    meta = meta[meta["view"].notna()]
    flow["frontal"] = int(len(meta))

    keys = cohort.sort_index().reset_index()[["stay_id", "subject_id", "intime"]].copy()
    keys["subject_id"] = keys["subject_id"].astype("int64")
    keys["intime"] = pd.to_datetime(keys["intime"])
    keys["t0"] = keys["intime"] + pd.Timedelta(hours=hours)
    keys["window_start"] = keys["intime"] - pd.Timedelta(hours=lookback_hours)
    meta = meta[meta["subject_id"].isin(keys["subject_id"])]
    flow["frontal_of_cohort_patients"] = int(len(meta))

    cand = meta[["dicom_id", "subject_id", "study_id", "study_datetime", "view"]].merge(
        keys, on="subject_id", how="inner"
    )
    cand = cand[(cand["study_datetime"] >= cand["window_start"]) & (cand["study_datetime"] <= cand["t0"])]
    flow["stay_image_pairs_in_window"] = int(len(cand))
    cand = cand.assign(view_rank=cand["view"].map(VIEW_RANK))
    cand = cand.sort_values(
        ["stay_id", "study_datetime", "view_rank", "dicom_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    chosen = cand.drop_duplicates(subset="stay_id", keep="first")

    paired = keys[["stay_id", "subject_id", "t0"]].merge(
        chosen[["stay_id", "study_id", "dicom_id", "study_datetime", "view"]], on="stay_id", how="left"
    )
    paired["has_cxr"] = paired["dicom_id"].notna()
    paired["study_id"] = paired["study_id"].astype("Int64")
    paired["hours_before_t0"] = (paired["t0"] - paired["study_datetime"]).dt.total_seconds() / 3600.0
    paired = paired[PAIRED_COLUMNS].sort_values("stay_id").reset_index(drop=True)
    flow["stays"] = int(len(paired))
    flow["stays_with_cxr"] = int(paired["has_cxr"].sum())
    return Linkage(paired=paired, flow=flow, candidates=cand.reset_index(drop=True))


def relative_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """MIMIC-CXR-JPG path: ``files/p10/p10000032/s50414267/<dicom_id>.jpg``."""
    subject = str(int(subject_id))
    return f"files/p{subject[:2]}/p{subject}/s{int(study_id)}/{dicom_id}.jpg"


def image_paths(paired: pd.DataFrame) -> list[str]:
    """Sorted, unique relative paths of the selected images only."""
    linked = paired[paired["has_cxr"]]
    paths = {
        relative_path(s, st, d)
        for s, st, d in zip(linked["subject_id"], linked["study_id"], linked["dicom_id"], strict=True)
    }
    return sorted(paths)


def write_image_list(paired: pd.DataFrame, path: str | Path) -> int:
    """Write the download list (one relative path per line); returns the number of images."""
    paths = image_paths(paired)
    Path(path).write_text("".join(f"{p}\n" for p in paths), encoding="utf-8", newline="\n")
    return len(paths)
