"""Synthetic MIMIC-CXR metadata keyed to a cohort, with planted cases and their answers.

No open X-ray data links to the demo patients, so Stage 1 is built and tested on fake
metadata with the real columns. Every cohort stay gets one planted scenario
(:data:`SCENARIOS`, assigned in turn) whose correct pairing is fixed when the rows are
written, never computed by the selection code. Patients with two cohort stays whose
windows come close get no images, so one stay's planted images can never land in the
other's window. Images of patients outside the cohort are added too and must be ignored.

All values are random or derived from the cohort's (already shifted) ICU times; nothing
comes from MIMIC-CXR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fate_icu.data.cxr.metadata import METADATA_COLUMNS

SCENARIOS = (
    "no_images",
    "single_ap_inside",
    "latest_of_several",
    "before_lookback_only",
    "at_lookback_edge",
    "at_t0_edge",
    "after_t0_only",
    "lateral_newer",
    "ap_pa_tie",
    "ap_ap_tie",
    "pa_only",
    "midnight",
    "small_time_value",
    "leading_zeros_lost",
    "no_fraction_text",
    "fraction_breaks_tie",
    "missing_view",
    "unparseable_time",
)
EXPECTED_COLUMNS = ["scenario", "has_cxr", "dicom_id", "study_id", "view", "study_datetime"]

_DESCRIPTION = {
    "AP": "CHEST (PORTABLE AP)",
    "PA": "CHEST (PA AND LAT)",
    "LATERAL": "CHEST (PA AND LAT)",
    "LL": "CHEST (PA AND LAT)",
}
_VIEW_CODE = {"AP": "antero-posterior", "PA": "postero-anterior", "LATERAL": "lateral", "LL": "left lateral"}
_ORIENTATION = {"AP": "Recumbent", "PA": "Erect"}


@dataclass
class SyntheticCxr:
    """Fake metadata (text columns, as read from the real file) and the planted answers."""

    metadata: pd.DataFrame
    expected: pd.DataFrame  # index stay_id; columns EXPECTED_COLUMNS

    def write_csv(self, path: str | Path) -> Path:
        """Write the metadata like the real file (gzip when the name ends in .gz)."""
        path = Path(path)
        self.metadata.to_csv(path, index=False)
        return path


def study_time_text(when: pd.Timestamp) -> str:
    """``StudyTime`` as MIMIC-CXR stores it: HHMMSS as a number, so ``00:15:20`` -> ``1520.0``."""
    whole = when.hour * 10_000 + when.minute * 100 + when.second
    fraction = f"{when.microsecond:06d}".rstrip("0") if when.microsecond else "0"
    return f"{whole}.{fraction}"


class _Rows:
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.rows: list[dict] = []
        self._dicoms: set[str] = set()
        self._studies: set[int] = set()

    def dicom(self) -> str:
        while True:
            value = "-".join(f"{self.rng.randint(0, 2**32, dtype=np.int64):08x}" for _ in range(5))
            if value not in self._dicoms:
                self._dicoms.add(value)
                return value

    def study(self) -> int:
        while True:
            value = int(self.rng.randint(50_000_000, 60_000_000))
            if value not in self._studies:
                self._studies.add(value)
                return value

    def add(
        self,
        subject: int,
        when: pd.Timestamp,
        view: str | None,
        *,
        study: int | None = None,
        dicom: str | None = None,
        time_text: str | None = None,
    ) -> tuple[str, int]:
        dicom = dicom or self.dicom()
        study = study or self.study()
        self.rows.append(
            {
                "dicom_id": dicom,
                "subject_id": str(subject),
                "study_id": str(study),
                "PerformedProcedureStepDescription": _DESCRIPTION.get(view, np.nan),
                "ViewPosition": view if view is not None else np.nan,
                "Rows": "3056",
                "Columns": "2544",
                "StudyDate": when.strftime("%Y%m%d"),
                "StudyTime": time_text if time_text is not None else study_time_text(when),
                "ProcedureCodeSequence_CodeMeaning": _DESCRIPTION.get(view, np.nan),
                "ViewCodeSequence_CodeMeaning": _VIEW_CODE.get(view, np.nan),
                "PatientOrientationCodeSequence_CodeMeaning": _ORIENTATION.get(view, np.nan),
            }
        )
        return dicom, study


def _first_clock_time(
    start: pd.Timestamp, end: pd.Timestamp, hour: int, minute: int, second: int, microsecond: int = 0
) -> pd.Timestamp | None:
    """Earliest time with this clock reading in [start, end]."""
    when = start.normalize() + pd.Timedelta(
        hours=hour, minutes=minute, seconds=second, microseconds=microsecond
    )
    while when < start:
        when += pd.Timedelta(days=1)
    return when if when <= end else None


def _conflicting_stays(keys: pd.DataFrame, lookback: pd.Timedelta, hours: pd.Timedelta) -> set[int]:
    """Stays of one patient whose planted image span could reach another stay's window."""
    out: set[int] = set()
    for _, group in keys.groupby("subject_id"):
        if len(group) < 2:
            continue
        rows = list(group.itertuples(index=False))
        for a in rows:
            span = (a.intime - lookback - pd.Timedelta(hours=1), a.intime + hours + pd.Timedelta(hours=6))
            for b in rows:
                if a.stay_id == b.stay_id:
                    continue
                window = (b.intime - lookback, b.intime + hours)
                if span[0] <= window[1] and window[0] <= span[1]:
                    out.update({a.stay_id, b.stay_id})
    return out


def generate_cxr_metadata(
    cohort: pd.DataFrame, hours: int = 24, lookback_hours: float = 24.0, seed: int = 0
) -> SyntheticCxr:
    """Fake ``mimic-cxr-2.0.0-metadata`` rows for a cohort, with the planted answers.

    Args:
        cohort: Index ``stay_id``; columns ``subject_id`` and ``intime``.
        hours: The window W (t0 = intime + W). Must be at least 21 h for every scenario.
        lookback_hours: The lookback the answers assume.
        seed: Random seed; equal seeds give identical metadata.
    """
    if hours < 21:
        raise ValueError("the planted scenarios assume a window of at least 21 hours")
    rng = np.random.RandomState(seed)
    rows = _Rows(rng)
    keys = cohort.sort_index().reset_index()[["stay_id", "subject_id", "intime"]].copy()
    keys["intime"] = pd.to_datetime(keys["intime"])
    w, lookback = pd.Timedelta(hours=hours), pd.Timedelta(hours=lookback_hours)
    conflicted = _conflicting_stays(keys, lookback, w)
    minute = pd.Timedelta(minutes=1)

    def h(x: float) -> pd.Timedelta:
        return pd.Timedelta(hours=x)

    def pick(subject: int, when: pd.Timestamp, view: str | None, **kw):
        dicom, study = rows.add(subject, when, view, **kw)
        return dicom, study, view, when

    expected = []
    for i, st in enumerate(keys.itertuples(index=False)):
        subject, intime = int(st.subject_id), st.intime
        t0, start = intime + w, intime - lookback
        scenario = "no_images" if st.stay_id in conflicted else SCENARIOS[i % len(SCENARIOS)]
        answer: tuple[str, int, str, pd.Timestamp] | None = None  # dicom, study, view, time

        if scenario == "single_ap_inside":
            answer = pick(subject, intime + h(6), "AP")
        elif scenario == "latest_of_several":
            pick(subject, start + (intime - start) / 2 if lookback_hours > 0 else intime + h(1), "AP")
            pick(subject, intime + h(3), "PA")
            answer = pick(subject, intime + h(20), "AP")
        elif scenario == "before_lookback_only":
            pick(subject, start - minute, "AP")
        elif scenario == "at_lookback_edge":
            answer = pick(subject, start, "AP")
        elif scenario == "at_t0_edge":
            answer = pick(subject, t0, "AP")
            pick(subject, t0 + minute, "AP")
        elif scenario == "after_t0_only":
            pick(subject, t0 + minute, "AP")
            pick(subject, t0 + h(5), "PA")
        elif scenario == "lateral_newer":
            answer = pick(subject, intime + h(2), "AP")
            pick(subject, intime + h(10), "LATERAL")
            pick(subject, intime + h(11), "LL")
        elif scenario == "ap_pa_tie":
            first, second = sorted([rows.dicom(), rows.dicom()])
            when, study = intime + h(8), rows.study()
            rows.add(subject, when, "PA", study=study, dicom=first)  # PA has the smaller dicom_id
            rows.add(subject, when, "AP", study=study, dicom=second)
            answer = (second, study, "AP", when)
        elif scenario == "ap_ap_tie":
            first, second = sorted([rows.dicom(), rows.dicom()])
            when, study = intime + h(9), rows.study()
            rows.add(subject, when, "AP", study=study, dicom=second)
            rows.add(subject, when, "AP", study=study, dicom=first)
            answer = (first, study, "AP", when)
        elif scenario == "pa_only":
            answer = pick(subject, intime + h(12), "PA")
        elif scenario in ("midnight", "small_time_value", "leading_zeros_lost", "no_fraction_text"):
            clock, text = {
                "midnight": ((0, 0, 0), "0.0"),
                "small_time_value": ((0, 0, 45), "45.0"),
                "leading_zeros_lost": ((0, 15, 20), "1520.0"),
                "no_fraction_text": ((9, 30, 0), "93000"),
            }[scenario]
            when = _first_clock_time(start, t0 - minute, *clock)
            if when is not None:  # (a window shorter than a day may miss the clock time)
                answer = pick(subject, when, "AP", time_text=text)
        elif scenario == "fraction_breaks_tie":
            when = _first_clock_time(start, t0 - minute, 14, 32, 10)
            if when is not None:
                later = when + pd.Timedelta(microseconds=123_000)
                first, second = sorted([rows.dicom(), rows.dicom()])
                rows.add(subject, when, "AP", dicom=first, time_text="143210.0")
                dicom, study = rows.add(subject, later, "AP", dicom=second, time_text="143210.123")
                answer = (dicom, study, "AP", later)  # truncating the fraction would pick the other
        elif scenario == "missing_view":
            answer = pick(subject, intime + h(4), "AP")
            pick(subject, intime + h(15), None)
        elif scenario == "unparseable_time":
            answer = pick(subject, intime + h(5), "AP")
            pick(subject, intime + h(16), "AP", time_text="246000.0")

        record = {
            "stay_id": st.stay_id,
            "scenario": scenario,
            "has_cxr": answer is not None,
            "dicom_id": np.nan,
            "study_id": pd.NA,
            "view": np.nan,
            "study_datetime": pd.NaT,
        }
        if answer is not None:
            record.update(dicom_id=answer[0], study_id=answer[1], view=answer[2], study_datetime=answer[3])
        expected.append(record)

    # images of patients outside the cohort, at times close to cohort windows: always ignored
    outsiders = sorted(set(range(19_999_000, 19_999_050)) - set(keys["subject_id"].astype(int)))[:10]
    for subject in outsiders:
        anchor = keys["intime"].iloc[int(rng.randint(0, len(keys)))]
        rows.add(subject, anchor + h(float(rng.uniform(-20, 20))).round("s"), "AP")

    metadata = pd.DataFrame(rows.rows, columns=METADATA_COLUMNS)
    metadata = metadata.iloc[rng.permutation(len(metadata))].reset_index(drop=True)
    expected_frame = pd.DataFrame(expected).set_index("stay_id")[EXPECTED_COLUMNS]
    expected_frame["study_id"] = expected_frame["study_id"].astype("Int64")
    expected_frame["study_datetime"] = pd.to_datetime(expected_frame["study_datetime"])
    return SyntheticCxr(metadata=metadata, expected=expected_frame)
