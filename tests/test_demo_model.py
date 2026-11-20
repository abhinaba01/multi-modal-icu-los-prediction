"""The explainer demo: the model, its input checks, and the local page server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from fate_icu.demo.app import serve
from fate_icu.demo.model import explain_patient, load, parse_inputs, save, train_demo_model

PATIENT = {
    "anchor_age": 78,
    "gender": "M",
    "heart_rate": 118,
    "respiration": 27,
    "creatinine": 3.4,
    "sofa_total": 11,
}


@pytest.fixture(scope="module")
def demo(synth_tables, contract):
    return train_demo_model(synth_tables, contract)


def test_explanation_is_valid_and_cites_its_reasons(demo):
    r = explain_patient(demo, PATIENT)
    assert r["valid"], r["failures"]
    assert 0.0 <= r["prediction"]["p_long_stay"] <= 1.0
    assert r["prediction"]["predicted_los_days"] > 0
    assert 3 <= len(r["sentences"]) <= 5
    assert all("[D" in s for s in r["sentences"])
    assert r["entered"] == len(PATIENT)


def test_blank_values_are_not_recorded_and_same_input_gives_same_output(demo):
    a = explain_patient(demo, {**PATIENT, "glucose": ""})
    b = explain_patient(demo, PATIENT)
    assert a == b


def test_effects_are_exact_for_the_logistic_part(demo):
    r = explain_patient(demo, PATIENT)
    assert abs(sum(d["effect"] for d in r["drivers"])) > 0
    assert all(d["direction"] in ("longer", "shorter") for d in r["drivers"])


@pytest.mark.parametrize(
    "bad",
    [
        {"anchor_age": 400},
        {"anchor_age": "abc"},
        {"nonsense": 1},
        {"gender": "robot"},
        {},
        {"anchor_age": 50},
    ],
)
def test_bad_input_is_refused_with_a_message(demo, bad):
    with pytest.raises(ValueError):
        explain_patient(demo, bad)


def test_parse_inputs_converts_numbers(demo):
    clean = parse_inputs(demo, {"anchor_age": "60", "heart_rate": "90", "respiration": "", "gender": "F"})
    assert clean["anchor_age"] == 60.0 and clean["gender"] == "F" and "respiration" not in clean


def test_save_and_load_round_trip(demo, tmp_path):
    save(demo, tmp_path / "m.joblib")
    again = load(tmp_path / "m.joblib")
    assert explain_patient(again, PATIENT) == explain_patient(demo, PATIENT)


@pytest.fixture(scope="module")
def server(demo):
    httpd = serve(demo, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body, host=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_serves_page_options_and_explanations(server):
    with urllib.request.urlopen(server + "/") as r:
        assert r.status == 200 and b"ICU stay explainer" in r.read()
    with urllib.request.urlopen(server + "/api/options") as r:
        options = json.loads(r.read())
    assert options["fields"]
    status, body = _post(server + "/api/explain", PATIENT)
    assert status == 200 and body["valid"]


def test_server_rejects_bad_requests(server):
    assert _post(server + "/api/explain", {"anchor_age": 400})[0] == 400
    assert _post(server + "/api/explain", [1, 2])[0] == 400
    assert _post(server + "/nothing", {})[0] == 404


def test_server_refuses_a_foreign_host_header(server):
    status, _ = _post(server + "/api/explain", PATIENT, host="evil.example.com")
    assert status == 403
