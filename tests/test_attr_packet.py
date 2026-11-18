"""Stage 4 evidence packet: the schema and the deterministic observed-text templates."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from fate_icu.attribution import packet as P
from fate_icu.data.contract import load_contract
from fate_icu.data.timeseries.variables import load_variables
from fate_icu.rag.drivers import driver_keys
from fate_icu.utils.env import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rag" / "packets"


def driver(i=1, effect=0.4, **kw):
    base = dict(id=f"D{i}", modality="timeseries", key="creatinine", label="serum creatinine",
                time="hours 12-24", observed="rose", direction="longer", effect=effect)  # fmt: skip
    return {**base, **kw}


def packet(drivers):
    return {
        "prediction": {
            "p_long_stay": 0.7,
            "predicted_los_days": 5.2,
            "long_stay_threshold_days": 4.0,
            "window_hours": 24,
        },
        "drivers": drivers,
    }


@pytest.mark.parametrize("name", ["packet_long.json", "packet_short.json"])
def test_the_stage_5_fixture_packets_are_valid_stage_4_packets(name):
    p = P.EvidencePacket.model_validate(json.loads((FIXTURES / name).read_text(encoding="utf-8")))
    assert 1 <= len(p.drivers) <= P.MAX_DRIVERS
    allowed: dict[str, set[str]] = {}
    for k in driver_keys():
        allowed.setdefault(k.modality, set()).add(k.key)
    P.check_keys(p, allowed)


def test_a_packet_round_trips_through_json():
    p = P.EvidencePacket.model_validate(
        packet([driver(1, 0.5), driver(2, 0.3, key="heart_rate", time="hours 0-4")])
    )
    again = P.EvidencePacket.model_validate_json(p.model_dump_json())
    assert again == p and again.schema_version == P.SCHEMA_VERSION


@pytest.mark.parametrize(
    "bad",
    [
        packet([driver(1, 0.3), driver(2, 0.5, key="heart_rate")]),  # not sorted by the size of the effect
        packet([driver(2, 0.5)]),  # ids must start at D1
        packet([driver(1, 0.5), driver(1, 0.4, key="heart_rate")]),  # repeated id
        packet([driver(1, 0.5), driver(2, 0.4)]),  # the same driver twice
        packet([driver(i, 0.9 - 0.1 * i, key=f"k{i}") for i in range(1, 7)]),  # six drivers
        packet([driver(1, effect=-0.2)]),  # a negative effect cannot be 'longer'
        packet([driver(1, effect=0.2, direction="shorter")]),
        packet([driver(1, effect=0.0)]),
        packet([driver(1, direction="up")]),
        packet([driver(1, modality="static", time="hours 0-4")]),  # a time on a static driver
        packet([driver(1, region="right lower lung zone")]),  # a region on a time-series driver
        packet([driver(1, reported_finding=True)]),
        packet([driver(1, label="")]),
        {
            **packet([driver(1)]),
            "prediction": {
                "p_long_stay": 1.4,
                "predicted_los_days": 5.0,
                "long_stay_threshold_days": 4.0,
                "window_hours": 24,
            },
        },
        {**packet([driver(1)]), "extra_field": 1},
    ],
)
def test_malformed_packets_are_refused(bad):
    with pytest.raises(ValidationError):
        P.EvidencePacket.model_validate(bad)


def test_an_unknown_key_is_refused_by_the_key_check():
    p = P.EvidencePacket.model_validate(packet([driver(1, key="made_up")]))
    with pytest.raises(ValueError, match="not a timeseries key"):
        P.check_keys(p, {"timeseries": {"creatinine"}})


def test_numbers_are_short_and_stable():
    assert [P.format_number(x) for x in (1.10, 2.345, 98.2, 98.7, 1234.5, 0.004567, 0.0, -3.14159, 10.0)] == [
        "1.1", "2.35", "98.2", "98.7", "1234", "0.0046", "0", "-3.14", "10",
    ]  # fmt: skip
    assert P.format_number(float("nan")) == P.NOT_RECORDED


@pytest.fixture(scope="module")
def creatinine():
    return next(v for v in load_variables().variables if v.name == "creatinine")


def raw_series(values):
    raw = np.full(24, np.nan)
    for hour, v in values.items():
        raw[hour] = v
    return raw


def test_a_rise_is_described_against_the_earlier_hours(creatinine):
    text = P.describe_timeseries(creatinine, raw_series({0: 1.0, 2: 1.2, 12: 2.2, 14: 2.4}), 12, 16)
    assert (
        text.startswith("rose from 1.1 to 2.3")
        and "above the reference range" in text
        and creatinine.unit in text
    )


def test_a_fall_a_steady_value_and_a_missing_block(creatinine):
    assert P.describe_timeseries(creatinine, raw_series({0: 2.0, 13: 1.0}), 12, 16).startswith(
        "fell from 2 to 1"
    )
    assert P.describe_timeseries(creatinine, raw_series({0: 1.0, 13: 1.05}), 12, 16).startswith(
        "stayed near 1.05"
    )
    assert P.describe_timeseries(creatinine, raw_series({0: 1.0}), 12, 16) == "not measured in this period"


def test_a_block_with_nothing_before_it_is_an_average(creatinine):
    text = P.describe_timeseries(creatinine, raw_series({1: 0.5, 2: 0.7}), 0, 4)
    assert text.startswith("averaged 0.6") and "below the reference range" not in text.replace(
        "0.6", ""
    )  # 0.6 is the lower limit
    assert "reference" not in P.describe_timeseries(creatinine, raw_series({1: 0.9}), 0, 4)


def test_the_same_input_always_gives_the_same_text(creatinine):
    raw = raw_series({0: 1.0, 13: 1.6})
    assert P.describe_timeseries(creatinine, raw, 12, 16) == P.describe_timeseries(
        creatinine, raw.copy(), 12, 16
    )


def test_static_groups_list_up_to_three_features_with_their_labels():
    contract = load_contract()
    vitals = [f.name for f in contract.features if f.group == "vitals"]
    values = {name: float(i) + 0.5 for i, name in enumerate(vitals)}
    values[vitals[1]] = float("nan")
    text = P.describe_static(contract, "vitals", values)
    parts = text.split("; ")
    assert len(parts) == 3 and parts[0].startswith(contract.feature(vitals[0]).label)
    assert parts[1].endswith(P.NOT_RECORDED)
    assert P.describe_static(contract, "vitals", {}) == P.NOT_RECORDED
    admission = [f.name for f in contract.features if f.group == "admission"][0]
    assert P.describe_static(contract, "admission", {admission: "MICU"}).endswith(": MICU")


def test_image_labels_never_state_a_diagnosis():
    assert P.describe_image("Effusion", 0.81, True) == "pleural effusion (image model estimate 0.81)"
    below = P.describe_image("Effusion", 0.31, False)
    assert "below its reporting threshold" in below and below.startswith("image model signal")
    assert P.describe_image("whole_image", None, False) == "chest X-ray (overall appearance, image model)"
    assert P.describe_image("Edema", None, False) == "image model signal for pulmonary edema"
    assert set(P.CONCEPT_NAMES) >= {k.key for k in driver_keys() if k.modality == "image"}


def test_a_shorter_driver_has_a_negative_effect_and_is_sorted_by_its_size():
    p = P.EvidencePacket.model_validate(
        packet([driver(1, -0.6, direction="shorter"), driver(2, 0.4, key="heart_rate", time="hours 0-4")])
    )
    assert [d.direction for d in p.drivers] == ["shorter", "longer"] and p.drivers[0].effect == -0.6
