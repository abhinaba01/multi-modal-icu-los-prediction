"""MIMIC-CXR-JPG metadata: the real columns, reading, and StudyDate/StudyTime parsing.

``StudyDate`` is ``YYYYMMDD``. ``StudyTime`` is ``HHMMSS`` with optional fractional
seconds, stored as a number, so leading zeros are lost: ``1520.0`` means 00:15:20.
Parsing works on the text. The whole-second part is zero-padded to six digits and the
fraction is kept to microseconds (MedFuse truncates it, which turns images a fraction
of a second apart into ties). Anything invalid becomes NaT and is counted, never
guessed. The metadata is read with every column as text so nothing is altered first.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
# Column order of mimic-cxr-2.0.0-metadata.csv: the identifiers, then the ten DICOM
# fields documented on the PhysioNet page. The lab run checks the real header.
METADATA_COLUMNS = [
    "dicom_id",
    "subject_id",
    "study_id",
    "PerformedProcedureStepDescription",
    "ViewPosition",
    "Rows",
    "Columns",
    "StudyDate",
    "StudyTime",
    "ProcedureCodeSequence_CodeMeaning",
    "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning",
]
REQUIRED_COLUMNS = ["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"]

_DATE = re.compile(r"^\d{8}$")
_TIME = re.compile(r"^(\d{1,6})(?:\.(\d*))?$")


def _as_text(value: Any) -> str | None:
    """The value as stripped text, or None when missing (NaN, None, empty)."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return None
        return repr(float(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    text = str(value).strip()
    return text or None


def parse_study_datetime(study_date: Any, study_time: Any) -> pd.Timestamp:
    """Combine one ``StudyDate`` and ``StudyTime`` into a timestamp (NaT if invalid).

    Examples:
        ``("21800101", "1520.0")`` -> 2180-01-01 00:15:20;
        ``("21800101", "45.0")`` -> 00:00:45; ``("21800101", "143210.123")`` -> 14:32:10.123.
    """
    date, time = _as_text(study_date), _as_text(study_time)
    if date is None or time is None:
        return pd.NaT
    if date.endswith(".0"):  # a date read as a number
        date = date[:-2]
    if not _DATE.match(date):
        return pd.NaT
    match = _TIME.match(time)
    if not match:
        return pd.NaT
    whole = match.group(1).zfill(6)
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    hour, minute, second = int(whole[:2]), int(whole[2:4]), int(whole[4:])
    if hour > 23 or minute > 59 or second > 59:
        return pd.NaT
    try:
        return pd.Timestamp(
            year=int(date[:4]),
            month=int(date[4:6]),
            day=int(date[6:]),
            hour=hour,
            minute=minute,
            second=second,
            microsecond=int(fraction),
        )
    except ValueError:  # not a calendar date, e.g. 30 February
        return pd.NaT


def parse_study_datetimes(dates: Iterable[Any], times: Iterable[Any]) -> pd.Series:
    """Vectorised :func:`parse_study_datetime` (same rules, element by element)."""
    values = [parse_study_datetime(d, t) for d, t in zip(dates, times, strict=True)]
    return pd.Series(pd.to_datetime(values), dtype="datetime64[ns]")


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read ``mimic-cxr-2.0.0-metadata.csv(.gz)`` with every column as text.

    Raises:
        FileNotFoundError: The file is missing.
        ValueError: A required column is missing or an identifier is not numeric.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIMIC-CXR metadata not found at {path} (see docs/lab_runbook.md, Stage 1)")
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks required column(s) {missing}; found {list(frame.columns)}")
    return normalise_identifiers(frame)


def normalise_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    """``subject_id`` and ``study_id`` as int64 (fail loudly if not numeric)."""
    out = frame.copy()
    for column in ("subject_id", "study_id"):
        numeric = pd.to_numeric(out[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{int(numeric.isna().sum())} metadata rows have a non-numeric {column}")
        out[column] = numeric.astype("int64")
    out["dicom_id"] = out["dicom_id"].astype(str).str.strip()
    return out
