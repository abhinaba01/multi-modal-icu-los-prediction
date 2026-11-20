"""Stage 6 output contract and the deterministic template explanation."""

from __future__ import annotations

import pytest

from fate_icu.attribution.packet import EvidencePacket
from fate_icu.llm import template, validate

PREDICTION = {
    "p_long_stay": 0.684,
    "predicted_los_days": 5.2,
    "long_stay_threshold_days": 4.0,
    "window_hours": 24,
}


def driver(i, key, label, direction, observed, effect):
    return {
        "id": f"D{i}", "modality": "static", "key": key, "label": label, "observed": observed,
        "direction": direction, "effect": effect if direction == "longer" else -effect,
    }  # fmt: skip


DRIVERS = [
    driver(1, "sofa_static", "SOFA score (worst of first 24 h)", "longer", "SOFA total 9 (typical 5)", 0.9),
    driver(
        2, "labs", "laboratory results (mean of first 24 h)", "longer", "creatinine 2.3 (typical 0.9)", 0.7
    ),
    driver(3, "demographics", "demographics", "shorter", "age (years) 34 (typical 61)", 0.5),
    driver(4, "vitals", "vital signs (mean of first 24 h)", "shorter", "heart rate 72 (typical 90)", 0.3),
    driver(5, "admission", "admission details", "longer", "admission type URGENT", 0.2),
]


def packet(n=5):
    return EvidencePacket.model_validate({"prediction": PREDICTION, "drivers": DRIVERS[:n]})


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_the_template_always_meets_the_contract(n):
    text, result = template.explain_checked(packet(n))
    assert result.ok, (n, result.failures, text)
    assert validate.MIN_SENTENCES <= len(validate.split_sentences(text)) <= validate.MAX_SENTENCES


def test_the_template_says_what_the_packet_says_and_nothing_more():
    text = template.explain(packet(5))
    sentences = validate.split_sentences(text)
    assert sentences[0].startswith(
        "The model predicts an ICU stay of about 5.2 days; the probability of a long stay"
    )
    assert (
        "more than 4 days" in sentences[0]
        and "is 68%" in sentences[0]
        and "[D1]" in sentences[0]
        and "[D2]" in sentences[0]
    )
    assert "longer stay [D1]" in sentences[1] and "shorter stay [D3]" in sentences[3]
    assert "Smaller contributions" in sentences[4] and "[D4]" in sentences[4] and "[D5]" in sentences[4]
    assert template.explain(packet(5)) == text  # deterministic


def test_the_template_needs_a_driver():
    with pytest.raises(ValueError, match="at least one driver"):
        template.explain({"prediction": PREDICTION, "drivers": []})


def test_the_template_accepts_the_dict_form_too():
    assert template.explain(packet(3).model_dump()) == template.explain(packet(3))


GOOD = template.explain(packet(5))


def failures(text, n=5, passages=()):
    return set(validate.check(text, packet(n), passages).failures)


def test_a_good_text_passes_and_is_truthy():
    result = validate.check(GOOD, packet(5))
    assert result.ok and bool(result) and result.failures == ()


def test_the_sentence_count_is_three_to_five():
    sentences = validate.split_sentences(GOOD)
    assert "sentence_count" in failures(" ".join(sentences[:2]))
    assert "sentence_count" in failures(GOOD + " The model is confident [D1].")
    assert "sentence_count" not in failures(" ".join(sentences[:3]))


def test_every_sentence_must_cite_a_driver_or_a_passage():
    assert "citation_missing" in failures(GOOD.replace(" [D3]", ""))


def test_only_ids_that_exist_may_be_cited():
    assert "unknown_id" in failures(GOOD.replace("[D3]", "[D9]"))
    assert "unknown_id" in failures(GOOD.replace("[D3]", "[R1]"))  # no passages were given
    passage = {"rid": "R1", "text": "Raised creatinine is linked to longer stays."}
    assert "unknown_id" not in failures(GOOD.replace("[D3]", "[D3] [R1]"), passages=[passage])
    assert "unknown_id" in failures(GOOD.replace("[D3]", "[D3] [R2]"), passages=[passage])


def test_a_number_that_is_not_in_the_packet_is_caught():
    assert "number_not_in_packet" in failures(GOOD.replace("68%", "72%"))
    assert "number_not_in_packet" in failures(GOOD.replace("creatinine 2.3", "creatinine 3.1"))
    assert "number_not_in_packet" in failures(GOOD.replace("(typical 61)", "(typical 58)"))
    assert "number_not_in_packet" not in failures(GOOD.replace("68%", "68.4%").replace("5.2 days", "5 days"))


def test_the_prediction_may_be_printed_in_the_packets_rounded_forms_only():
    assert "number_not_in_packet" not in failures(GOOD.replace("about 5.2 days", "about 5.2 days"))
    assert "number_not_in_packet" in failures(GOOD.replace("about 5.2 days", "about 6.4 days"))


def test_the_first_sentence_must_state_the_prediction():
    sentences = validate.split_sentences(GOOD)
    sentences[0] = (
        "The model has looked at the patient and reached a view mainly because of the SOFA score [D1]."
    )
    assert "first_sentence_prediction" in failures(" ".join(sentences))


def test_a_driver_sentence_must_use_the_drivers_own_direction():
    assert "direction" in failures(GOOD.replace("longer stay [D1]", "shorter stay [D1]"))
    assert "direction" in failures(GOOD.replace("shorter stay [D3]", "longer stay [D3]"))
    assert "direction" in failures(GOOD.replace("shorter stay [D3]", "stay [D3]"))  # no direction word at all
    both = GOOD.replace("shorter stay [D3]", "shorter stay, not a longer one [D3]")
    assert "direction" in failures(both)


def test_a_sentence_citing_drivers_of_both_directions_must_name_both():
    text = template.explain(packet(5))
    assert "direction" not in failures(text)  # the closing sentence cites D4 (shorter) and D5 (longer)
    assert "direction" in failures(text.replace("toward a longer stay [D5]", "toward a shorter stay [D5]"))


def test_the_first_sentence_is_exempt_from_the_direction_rule():
    assert "direction" not in failures(GOOD)  # it cites D1 (longer) and D2 (longer) without the word


@pytest.mark.parametrize(
    "advice",
    [
        "The patient should be given antibiotics [D1].",
        "We recommend starting dialysis [D2].",
        "Clinicians should consider starting vasopressors [D1].",
        "Administer fluids early [D1].",
    ],
)
def test_treatment_advice_is_refused(advice):
    sentences = validate.split_sentences(GOOD)
    text = " ".join([*sentences[:3], advice])
    assert "treatment_advice" in failures(text)


def test_describing_a_value_is_not_treatment_advice():
    assert "treatment_advice" not in failures(GOOD)


def test_sentences_are_split_at_full_stops_not_inside_numbers_or_ranges():
    text = "The stay is 5.2 days [D1]. Creatinine 0.6-1.2 mg/dL is normal [D2]. It rose from 1.1 to 2.3 [D2]."
    assert validate.split_sentences(text) == [
        "The stay is 5.2 days [D1].",
        "Creatinine 0.6-1.2 mg/dL is normal [D2].",
        "It rose from 1.1 to 2.3 [D2].",
    ]


def test_every_failure_is_named_once_and_in_the_order_the_rules_run():
    broken = "The model has a view. It is 99 [D9]. Administer fluids."
    result = validate.check(broken, packet(5))
    assert not result.ok and len(result.failures) == len(set(result.failures))
    assert {"first_sentence_prediction", "number_not_in_packet", "citation_missing", "unknown_id",
            "treatment_advice"} <= set(result.failures)  # fmt: skip


def test_a_number_borrowed_from_another_driver_is_caught():
    # 2.3 is driver D2's creatinine: D3's sentence may not quote it, though the packet does contain it
    borrowed = GOOD.replace("age (years) 34 (typical 61)", "age (years) 34 (typical 2.3)")
    assert "number_not_in_packet" in failures(borrowed)
    # the same number is fine in D2's own sentence
    assert "number_not_in_packet" not in failures(GOOD)


def test_a_passage_number_is_allowed_only_where_the_passage_is_cited():
    passage = {"rid": "R1", "text": "Stays were 3.4 days longer with raised creatinine."}
    cited = GOOD.replace("creatinine 2.3 (typical 0.9)", "creatinine 2.3 (typical 0.9), 3.4 days")
    cited = cited.replace("longer stay [D2]", "longer stay [D2] [R1]")
    assert "number_not_in_packet" not in failures(cited, passages=[passage])
    uncited = GOOD.replace("age (years) 34 (typical 61)", "age (years) 34 (typical 61), 3.4 days")
    assert "number_not_in_packet" in failures(uncited, passages=[passage])
