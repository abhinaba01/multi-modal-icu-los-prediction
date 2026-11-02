"""MIMIC-CXR metadata: StudyDate/StudyTime parsing and reading the file as text."""

from __future__ import annotations

import pandas as pd
import pytest

from fate_icu.data.cxr.metadata import (
    METADATA_COLUMNS,
    parse_study_datetime,
    parse_study_datetimes,
    read_metadata,
)

T = pd.Timestamp


@pytest.mark.parametrize(
    "date, time, expected",
    [
        ("21800101", "0.0", T("2180-01-01 00:00:00")),  # midnight
        ("21800101", "45.0", T("2180-01-01 00:00:45")),  # below 100: seconds only
        ("21800101", "1520.0", T("2180-01-01 00:15:20")),  # leading zeros lost
        ("21800101", "93000", T("2180-01-01 09:30:00")),  # no fraction at all
        ("21800101", "143210.123", T("2180-01-01 14:32:10.123")),  # fraction kept
        ("21800101", "143210.1234567", T("2180-01-01 14:32:10.123456")),  # beyond microseconds: truncated
        ("21800101", "235959.999999", T("2180-01-01 23:59:59.999999")),
        (21800101, 1520.0, T("2180-01-01 00:15:20")),  # numbers, as a default CSV reader gives them
        ("21800101.0", 0.0, T("2180-01-01 00:00:00")),
        (" 21800101 ", " 1520.0 ", T("2180-01-01 00:15:20")),
    ],
)
def test_valid_study_datetimes(date, time, expected):
    assert parse_study_datetime(date, time) == expected


@pytest.mark.parametrize(
    "date, time",
    [
        ("21800101", "246000.0"),  # hour 24
        ("21800101", "126000"),  # minute 60
        ("21800101", "125960"),  # second 60
        ("21800101", "1234567"),  # seven digits
        ("21800101", "12a000"),
        ("21800101", ""),
        ("21800101", None),
        ("21800101", float("nan")),
        ("2180010", "120000"),  # seven-digit date
        ("21800230", "120000"),  # 30 February
        (None, "120000"),
        ("2180-01-01", "120000"),
    ],
)
def test_invalid_values_become_nat(date, time):
    assert pd.isna(parse_study_datetime(date, time))


def test_vectorised_parsing_matches_the_scalar_rule():
    dates = ["21800101", "21800102", None, "21800230", "21991231"]
    times = ["1520.0", "93000", "120000", "120000", "235959.5"]
    out = parse_study_datetimes(dates, times)
    expected = [parse_study_datetime(d, t) for d, t in zip(dates, times, strict=True)]
    assert out.dtype == "datetime64[ns]"
    assert [x if pd.notna(x) else None for x in out] == [x if pd.notna(x) else None for x in expected]


def test_metadata_is_read_as_text_and_checked(cxr_synth, tmp_path):
    path = cxr_synth.write_csv(tmp_path / "mimic-cxr-2.0.0-metadata.csv.gz")
    frame = read_metadata(path)
    assert list(frame.columns) == METADATA_COLUMNS
    assert {"1520.0", "45.0", "93000", "143210.123"} <= set(frame["StudyTime"].dropna())  # untouched text
    assert frame["subject_id"].dtype == "int64" and frame["study_id"].dtype == "int64"
    bad = tmp_path / "bad.csv"
    cxr_synth.metadata.drop(columns=["ViewPosition"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="ViewPosition"):
        read_metadata(bad)
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "missing.csv.gz")
