"""Explanation faithfulness metrics, each on an example worked out by hand."""

from __future__ import annotations

import random

import pytest

from fate_icu.attribution.packet import EvidencePacket
from fate_icu.eval.explanation import (
    Entity,
    Lexicon,
    add_decoy,
    build_lexicon,
    claim_deletion_validity,
    direction_accuracy,
    driver_precision,
    driver_recall_at_k,
    drop_driver,
    numeric_consistency,
    packet_sensitivity,
    per_stay_metrics,
    random_decoy,
    stated_directions,
)
from fate_icu.llm.template import explain

HR, LABS, SOFA = Entity("timeseries", "heart_rate"), Entity("static", "labs"), Entity("static", "sofa_static")
PACKET = {
    "prediction": {
        "p_long_stay": 0.7,
        "predicted_los_days": 6.2,
        "long_stay_threshold_days": 4.1,
        "window_hours": 24,
    },
    "drivers": [
        {
            "id": "D1",
            "modality": "timeseries",
            "key": "heart_rate",
            "label": "heart rate",
            "time": "hours 12-16",
            "observed": "rose from 82 to 118 bpm, above the reference range",
            "direction": "longer",
            "effect": 1.4,
        },
        {
            "id": "D2",
            "modality": "static",
            "key": "labs",
            "label": "laboratory results (mean of first 24 h)",
            "observed": "mean serum creatinine (mg/dL): 0.8",
            "direction": "shorter",
            "effect": -0.9,
        },
        {
            "id": "D3",
            "modality": "static",
            "key": "sofa_static",
            "label": "SOFA score (worst of first 24 h)",
            "observed": "total SOFA score: 11",
            "direction": "longer",
            "effect": 0.5,
        },
    ],
}


@pytest.fixture(scope="module")
def lexicon():
    return build_lexicon(concepts=["Effusion", "Edema"])


def test_the_packet_used_here_is_valid():
    EvidencePacket.model_validate(PACKET)


def test_matching_is_longest_first_and_a_span_is_used_once(lexicon):
    text = (
        "Mean heart rate (beats/min) was high and the heart rate rose; "
        "a pleural effusion is on the chest X-ray."
    )
    found = [(e.modality, e.key) for e, _, _ in lexicon.find(text)]
    assert found == [
        ("static", "vitals"),
        ("timeseries", "heart_rate"),
        ("image", "Effusion"),
        ("image", "whole_image"),
    ]


def test_matching_is_case_insensitive_and_whole_word_only(lexicon):
    assert Entity("timeseries", "heart_rate") in lexicon.entities_in("HEART RATE fell")
    assert lexicon.entities_in("an apparatus of rates") == set()  # no unit is named inside other words


def test_a_phrase_shared_by_two_units_is_an_error():
    with pytest.raises(ValueError, match="belongs to both"):
        Lexicon({Entity("static", "a"): ["serum sodium"], Entity("timeseries", "b"): ["Serum  sodium"]})


def test_driver_precision_is_the_matched_share_of_named_entities(lexicon):
    text = "Heart rate [D1], laboratory results (mean of first 24 h) [D2] and a pleural effusion [D9] matter."
    assert driver_precision(text, PACKET, lexicon) == pytest.approx(2 / 3)  # names 3 units, 2 are drivers
    assert driver_precision("Nothing is named here.", PACKET, lexicon) is None


def test_driver_recall_at_k_uses_the_largest_effects(lexicon):
    text = "Heart rate is described. The SOFA score (worst of first 24 h) is described."
    assert driver_recall_at_k(text, PACKET, lexicon, k=3) == pytest.approx(2 / 3)  # D2 is missing
    assert driver_recall_at_k(text, PACKET, lexicon, k=1) == 1.0  # D1 (|1.4|) is the largest and is named
    assert driver_recall_at_k(text, {"drivers": []}, lexicon) is None


def test_stated_directions_follow_names_and_citations(lexicon):
    text = (
        "Heart rate points toward a longer stay. "
        "The laboratory results (mean of first 24 h) [D2] point toward a longer stay. "
        "Another sentence cites [D3] without a direction."
    )
    stated = stated_directions(text, PACKET, lexicon)
    assert stated == {"D1": {"longer"}, "D2": {"longer"}, "D3": set()}
    assert (
        direction_accuracy(text, PACKET, lexicon) == 0.5
    )  # D1 right, D2 wrong (it is "shorter"), D3 not stated


def test_a_driver_described_both_ways_is_wrong(lexicon):
    text = "Heart rate [D1] points toward a longer stay, not a shorter one."
    assert direction_accuracy(text, PACKET, lexicon) == 0.0


def test_numeric_consistency_counts_sentences_with_unsupported_numbers():
    text = (
        "The model predicts about 6.2 days with a 70% chance of a long stay. "
        "Heart rate rose from 82 to 118 bpm [D1]. "
        "The creatinine was 9.9 [D2]."
    )
    assert numeric_consistency(text, PACKET) == pytest.approx(2 / 3)  # 9.9 belongs to no cited driver
    assert numeric_consistency("", PACKET) is None


def test_claim_deletion_validity_needs_the_right_sign_and_size(lexicon):
    text = (
        "Heart rate [D1] points toward a longer stay. "
        "The laboratory results (mean of first 24 h) [D2] shorter stay."
    )
    delta = {"D1": 1.0, "D2": -0.05}  # D1 moves the logit up by 1.0, D2 down by only 0.05
    assert claim_deletion_validity(text, PACKET, lexicon, delta.__getitem__, tau=0.1) == 0.5
    assert claim_deletion_validity(text, PACKET, lexicon, delta.__getitem__, tau=0.01) == 1.0
    wrong = {"D1": -1.0, "D2": 0.5}  # both signs are opposite to what the text says
    assert claim_deletion_validity(text, PACKET, lexicon, wrong.__getitem__, tau=0.1) == 0.0
    assert claim_deletion_validity("Nothing is said.", PACKET, lexicon, delta.__getitem__, tau=0.1) is None


def test_drop_driver_renumbers_and_add_decoy_stays_valid():
    smaller = drop_driver(PACKET, "D1")
    assert [d["id"] for d in smaller["drivers"]] == ["D1", "D2"] and smaller["drivers"][0]["key"] == "labs"
    EvidencePacket.model_validate(smaller)
    decoy = {
        "modality": "timeseries",
        "key": "creatinine",
        "label": "serum creatinine",
        "time": "hours 8-12",
        "observed": "stayed near 0.8 mg/dL",
        "direction": "shorter",
    }
    bigger = add_decoy(PACKET, decoy)
    assert bigger["drivers"][-1]["id"] == "D4" and bigger["drivers"][-1]["effect"] < 0
    EvidencePacket.model_validate(bigger)  # sorted by |effect|, ids in order, no duplicate unit


def test_a_generator_that_follows_the_packet_passes_packet_sensitivity(lexicon):
    decoy = {
        "modality": "timeseries",
        "key": "creatinine",
        "label": "serum creatinine",
        "time": "hours 8-12",
        "observed": "stayed near 0.8 mg/dL",
        "direction": "shorter",
    }
    out = packet_sensitivity(PACKET, lexicon, explain, decoy)
    assert out == {"drop_followed": True, "decoy_followed": True}


def test_a_generator_that_ignores_the_packet_fails_packet_sensitivity(lexicon):
    decoy = {
        "modality": "timeseries",
        "key": "creatinine",
        "label": "serum creatinine",
        "direction": "shorter",
    }
    stubborn = lambda packet: "Heart rate [D1] points toward a longer stay."  # noqa: E731
    out = packet_sensitivity(PACKET, lexicon, stubborn, decoy)
    assert out == {"drop_followed": False, "decoy_followed": False}


def test_nothing_to_follow_when_the_original_text_did_not_name_the_dropped_driver(lexicon):
    decoy = {
        "modality": "timeseries",
        "key": "creatinine",
        "label": "serum creatinine",
        "direction": "shorter",
    }
    silent = lambda packet: "No unit is named in this text [D1]."  # noqa: E731
    assert packet_sensitivity(PACKET, lexicon, silent, decoy)["drop_followed"] is None


def test_per_stay_metrics_of_the_template_explanation(lexicon):
    metrics = per_stay_metrics(explain(PACKET), PACKET, [], lexicon)
    assert metrics["driver_precision"] == 1.0 and metrics["unattributed_rate"] == 0.0
    assert metrics["driver_recall_at_3"] == 1.0 and metrics["direction_accuracy"] == 1.0
    assert metrics["numeric_consistency"] == 1.0  # the template can only say what the packet says


def test_random_decoy_avoids_the_packets_own_units():
    pool = [
        {"modality": "timeseries", "key": "heart_rate", "label": "heart rate", "direction": "longer"},
        {"modality": "timeseries", "key": "lactate", "label": "lactate", "direction": "longer"},
    ]
    assert random_decoy(PACKET, pool, random.Random(0))["key"] == "lactate"
    with pytest.raises(ValueError):
        random_decoy(PACKET, pool[:1], random.Random(0))
