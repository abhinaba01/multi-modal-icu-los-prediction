"""Stage 5 queries: labels for every driver key, reasons for changes, no patient values."""

from __future__ import annotations

import pytest
import yaml

from fate_icu.rag.drivers import DriverKey, driver_keys
from fate_icu.rag.queries import (
    DEFAULT_QUERIES_PATH,
    build_queries,
    check_no_patient_values,
    load_query_spec,
    query_text,
)


@pytest.fixture(scope="module")
def spec():
    return load_query_spec()


def test_every_driver_key_has_one_label_in_driver_order(spec, variables, contract):
    keys = driver_keys(variables, contract)
    assert [(label.modality, label.key) for label in spec.labels] == [(k.modality, k.key) for k in keys]
    for label, key in zip(spec.labels, keys, strict=True):
        assert label.source_label == key.source_label
        assert label.label == key.source_label or label.reason


def test_queries_follow_the_template(spec):
    queries = build_queries(spec)
    assert len(queries) == 100 and len({q.query_id for q in queries}) == 100
    assert queries[0].query_id == "timeseries:heart_rate:longer" and queries[1].direction == "shorter"
    assert queries[0].text == "heart rate associated with longer intensive care unit length of stay"
    hb = query_text(spec, "timeseries", "haemoglobin", "shorter")
    assert hb == "hemoglobin associated with shorter intensive care unit length of stay"
    for q in queries:
        label = spec.label_for(q.modality, q.key).label
        assert q.text == spec.template.format(label=label, direction=q.direction)
        check_no_patient_values(q.text)
    with pytest.raises(ValueError):
        query_text(spec, "timeseries", "lactate", "sideways")
    with pytest.raises(KeyError):
        spec.label_for("timeseries", "not_a_channel")


@pytest.mark.parametrize(
    "text", ["oxygen saturation (SpO2)", "PaO2/FiO2 ratio", "first day of intensive care"]
)
def test_allowed_text(text):
    check_no_patient_values(text)


@pytest.mark.parametrize(
    "text", ["lactate above 4", "creatinine 2.3 mg/dL", "SOFA of 24 h", "saturation 90%", "1.5"]
)
def test_patient_values_are_rejected(text):
    with pytest.raises(ValueError, match="patient value"):
        check_no_patient_values(text)


def _write(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_QUERIES_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_spec_validation(tmp_path):
    def drop(raw):
        del raw["labels"]["timeseries"]["lactate"]

    with pytest.raises(ValueError, match=r"missing \['timeseries:lactate'\]"):
        load_query_spec(_write(tmp_path, drop))

    def extra(raw):
        raw["labels"]["timeseries"]["lactat"] = {"label": "lactate"}

    with pytest.raises(ValueError, match=r"extra \['timeseries:lactat'\]"):
        load_query_spec(_write(tmp_path, extra))

    def no_reason(raw):
        raw["labels"]["timeseries"]["heart_rate"] = {"label": "pulse"}

    with pytest.raises(ValueError, match="give a reason"):
        load_query_spec(_write(tmp_path, no_reason))

    def number(raw):
        raw["labels"]["timeseries"]["lactate"] = {"label": "lactate above 4", "reason": "x"}

    with pytest.raises(ValueError, match="patient value"):
        load_query_spec(_write(tmp_path, number))

    def template(raw):
        raw["template"] = "{label} and stay"

    with pytest.raises(ValueError, match="exactly"):
        load_query_spec(_write(tmp_path, template))

    def image_model(raw):
        raw["image_model"] = "densenet121-res224-all"

    with pytest.raises(ValueError, match="image model"):
        load_query_spec(_write(tmp_path, image_model))


def test_sha256_is_stable_and_sensitive(spec, tmp_path):
    assert load_query_spec().sha256 == spec.sha256

    def change(raw):
        raw["labels"]["timeseries"]["lactate"] = {"label": "blood lactate", "reason": "test"}

    assert load_query_spec(_write(tmp_path, change)).sha256 != spec.sha256


def test_custom_keys():
    keys = [DriverKey("timeseries", "lactate", "lactate")]
    with pytest.raises(ValueError, match="extra"):
        load_query_spec(keys=keys)  # the real file has 49 more keys
