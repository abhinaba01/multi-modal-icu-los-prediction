"""The aggregate flow report of the X-ray linkage."""

from __future__ import annotations

import pytest

from fate_icu.data.cxr.report import FLOW_ORDER, check_planted_answers, flow_report


@pytest.fixture(scope="module")
def report(cxr_linkage, cxr_synth, synth, synth_tables, synth_split):
    groups = synth.tables["patients"].set_index("subject_id")["anchor_year_group"]
    return flow_report(cxr_linkage, cxr_synth.metadata, synth_tables.cohort, synth_split, groups, 4.0, 24.0)


def test_flow_counts_only_decrease(cxr_linkage):
    sequence = [cxr_linkage.flow[k] for k in FLOW_ORDER]
    assert sequence == sorted(sequence, reverse=True)
    assert cxr_linkage.flow["stays_with_cxr"] <= cxr_linkage.flow["stays"]


def test_report_contents(report):
    assert set(report["linked_share_by_split"]) == {"train", "val", "test"}
    assert "2014 - 2016" in report["linked_share_by_anchor_year_group"]
    assert report["below_min_paired_train_stays"] is True  # a 60-patient fixture is far below 2,000
    assert set(report["los_days"]) == {"paired", "unpaired"}
    alignment = report["date_alignment"]
    assert set(alignment) == {"cohort_patients_with_any_xray", "share_with_an_xray_during_an_icu_stay"}
    for value in report["flow"].values():
        assert isinstance(value, str) or value >= 10  # small counts suppressed


def test_planted_check_catches_a_wrong_pairing(cxr_linkage, cxr_synth):
    broken = cxr_linkage.paired.copy()
    row = broken.index[broken["has_cxr"]][0]
    broken.loc[row, "dicom_id"] = "not-the-planted-image"
    result = check_planted_answers(broken, cxr_synth.expected)
    assert not result["all_correct"] and sum(result["wrong_by_scenario"].values()) == 1
