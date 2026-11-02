"""Stage 1 selection: every planted answer, the spec's output, and the pairing rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fate_icu.data.cxr import cache as cache_cli
from fate_icu.data.cxr import link as link_cli
from fate_icu.data.cxr.linkage import PAIRED_COLUMNS, frontal_view, link, relative_path, write_image_list
from fate_icu.data.cxr.report import check_planted_answers
from fate_icu.data.cxr.synthetic import SCENARIOS, generate_cxr_metadata
from fate_icu.tabular.loaders import FrameLoader
from fate_icu.utils.env import load_settings

H = 24


def test_every_planted_answer_is_correct(cxr_linkage, cxr_synth):
    result = check_planted_answers(cxr_linkage.paired, cxr_synth.expected)
    assert result["all_correct"], result["wrong_by_scenario"]
    assert result["scenarios_planted"] == len(SCENARIOS)  # every scenario is exercised


def test_answers_one_by_one(cxr_linkage, cxr_synth):
    got = cxr_linkage.paired.set_index("stay_id")
    exp = cxr_synth.expected
    assert (got["has_cxr"] == exp["has_cxr"]).all()
    linked = exp[exp["has_cxr"]]
    for column in ("dicom_id", "view", "study_datetime"):
        assert (got.loc[linked.index, column] == linked[column]).all(), column
    assert (got.loc[linked.index, "study_id"].astype("int64") == linked["study_id"].astype("int64")).all()


def test_scenario_rules(cxr_linkage, cxr_synth):
    got = cxr_linkage.paired.set_index("stay_id").join(cxr_synth.expected["scenario"])
    by = {name: frame for name, frame in got.groupby("scenario")}
    for name in ("no_images", "before_lookback_only", "after_t0_only"):
        assert not by[name]["has_cxr"].any(), name
    assert (by["at_t0_edge"]["hours_before_t0"] == 0).all()  # t0 itself is eligible
    assert (by["at_lookback_edge"]["hours_before_t0"] == H + 24).all()  # intime - 24 h is eligible
    assert (by["ap_pa_tie"]["view"] == "AP").all() and (by["pa_only"]["view"] == "PA").all()
    assert (by["fraction_breaks_tie"]["study_datetime"].dt.microsecond == 123_000).all()
    assert (by["lateral_newer"]["view"] == "AP").all()


def test_every_stay_is_kept_with_the_spec_columns(cxr_linkage, synth_tables):
    paired = cxr_linkage.paired
    assert list(paired.columns) == PAIRED_COLUMNS
    assert paired["stay_id"].tolist() == sorted(synth_tables.cohort.index)
    linked, unlinked = paired[paired["has_cxr"]], paired[~paired["has_cxr"]]
    assert len(linked) and len(unlinked)
    assert set(linked["view"]) <= {"AP", "PA"}
    t0 = synth_tables.cohort["t0"].reindex(linked["stay_id"]).to_numpy()
    hours = (t0 - linked["study_datetime"].to_numpy()) / np.timedelta64(1, "h")
    np.testing.assert_allclose(linked["hours_before_t0"].to_numpy(), hours)
    assert linked["hours_before_t0"].between(0, H + 24).all()
    assert unlinked["dicom_id"].isna().all() and unlinked["study_datetime"].isna().all()
    assert unlinked["hours_before_t0"].isna().all()


def test_result_ignores_row_order(cxr_synth, cxr_linkage, synth_tables):
    shuffled = cxr_synth.metadata.sample(frac=1.0, random_state=11)
    pd.testing.assert_frame_equal(link(shuffled, synth_tables.cohort, H, 24.0).paired, cxr_linkage.paired)


def test_lookback_is_configurable(cxr_synth, synth_tables):
    zero = generate_cxr_metadata(synth_tables.cohort, H, 0.0, seed=3)
    assert check_planted_answers(link(zero.metadata, synth_tables.cohort, H, 0.0).paired, zero.expected)[
        "all_correct"
    ]
    narrow = link(cxr_synth.metadata, synth_tables.cohort, H, 0.0).paired.set_index("stay_id")
    edge = cxr_synth.expected.index[cxr_synth.expected["scenario"] == "at_lookback_edge"]
    assert not narrow.loc[edge, "has_cxr"].any()  # the image at intime - 24 h no longer counts
    with pytest.raises(ValueError):
        link(cxr_synth.metadata, synth_tables.cohort, H, -1.0)


def test_labels_play_no_part(cxr_synth, cxr_linkage, synth_tables):
    """Real pairings only: shuffling LOS across stays changes nothing (CLAUDE.md A4)."""
    cohort = synth_tables.cohort.copy()
    cohort["los"] = np.random.RandomState(0).permutation(cohort["los"].to_numpy())
    pd.testing.assert_frame_equal(link(cxr_synth.metadata, cohort, H, 24.0).paired, cxr_linkage.paired)


def test_only_frontal_views_count():
    views = frontal_view(pd.Series(["AP", " pa ", "LATERAL", "LL", "AP AXIAL", None, np.nan]))
    assert views.iloc[:2].tolist() == ["AP", "PA"] and views.iloc[2:].isna().all()


def test_an_image_can_serve_two_stays_of_one_patient():
    cohort = pd.DataFrame(
        {
            "subject_id": [10000001, 10000001],
            "intime": [pd.Timestamp("2180-01-01 08:00"), pd.Timestamp("2180-01-02 20:00")],
        },
        index=pd.Index([1, 2], name="stay_id"),
    )
    metadata = pd.DataFrame(
        [
            {
                "dicom_id": "a" * 8 + "-" + "b" * 8 + "-" + "c" * 8 + "-" + "d" * 8 + "-" + "e" * 8,
                "subject_id": "10000001",
                "study_id": "50000001",
                "ViewPosition": "AP",
                "StudyDate": "21800101",
                "StudyTime": "220000.0",
            }
        ]
    )
    paired = link(metadata, cohort, H, 24.0).paired
    assert paired["has_cxr"].all() and paired["dicom_id"].nunique() == 1


def test_image_paths_and_download_list(cxr_linkage, tmp_path):
    assert relative_path(10000032, 50414267, "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014") == (
        "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    )
    n = write_image_list(cxr_linkage.paired, tmp_path / "image_list.txt")
    lines = (tmp_path / "image_list.txt").read_text(encoding="utf-8").splitlines()
    linked = cxr_linkage.paired[cxr_linkage.paired["has_cxr"]]
    assert n == len(lines) == linked["dicom_id"].nunique()
    assert lines == sorted(set(lines))
    assert all(line.startswith("files/p1") and line.endswith(".jpg") for line in lines)
    assert {Path(line).stem for line in lines} == set(linked["dicom_id"])  # the selected images only


# --- the two commands, on synthetic data ---------------------------------------------------


def test_link_command_writes_outputs(synth, synth_tables, contract, tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("FATE_DB_URL", "postgresql+psycopg://alice:s3cret@localhost:5432/mimiciv_demo")
    config = link_cli.CxrConfig(run_name="unit", metadata="synthetic", images="fixtures")
    report = link_cli.run(
        config, contract, tables=synth_tables, loader=FrameLoader(synth.tables), settings=load_settings()
    )
    out = tmp_path / "cxr" / report["run_id"]
    for name in ("paired_cohort.parquet", "image_list.txt", "report.json", "run_manifest.json"):
        assert (out / name).exists(), name
    assert report["planted_answers"]["all_correct"]
    assert len(pd.read_parquet(out / "paired_cohort.parquet")) == len(synth_tables.cohort)
    assert "s3cret" not in (out / "run_manifest.json").read_text(encoding="utf-8")
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["flow"]


def test_cache_command_on_the_public_images(tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_ARTIFACTS_DIR", str(tmp_path))
    config = link_cli.CxrConfig(run_name="unit", metadata="synthetic", images="fixtures")
    first = cache_cli.run(config, settings=load_settings())
    assert first["ok"] and first["counts"]["requested"] == "<10"  # 4 images, suppressed like any count
    assert len(list((tmp_path / "cxr_cache").rglob("*.npz"))) == 4
    second = cache_cli.run(config, settings=load_settings())
    assert second["ok"] and len(list((tmp_path / "cxr_cache").rglob("*.npz"))) == 4
