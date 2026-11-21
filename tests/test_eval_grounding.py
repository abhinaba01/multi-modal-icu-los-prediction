"""Groundedness metrics on hand-built claims, with the deterministic word-overlap checker."""

from __future__ import annotations

import math

import pytest

from fate_icu.eval.grounding import (
    LexicalChecker,
    NLIChecker,
    checker_agreement,
    claim_counts,
    cohens_kappa,
    rates,
    split_claims,
)

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
        }
    ],
}
R1 = {"rid": "R1", "text": "Higher organ failure scores predict longer ICU stays in many cohorts."}
R2 = {"rid": "R2", "text": "Elevated creatinine marks kidney injury and slower recovery."}
PATIENT = "Heart rate rose from 82 to 118 bpm [D1]."
GENERAL = "Higher organ failure scores predict longer ICU stays [R1]."
CHECK = LexicalChecker()


def counts(text, passages):
    return claim_counts(text, PACKET, passages, CHECK)


def test_claims_are_sentences_without_their_citation_tags():
    claims = split_claims("Heart rate rose [D1] [D2]. Scores predict stays [R1] [R2] [R1].")
    assert [c.text for c in claims] == ["Heart rate rose.", "Scores predict stays."]
    assert claims[0].driver_ids == ("D1", "D2") and claims[0].passage_ids == ()
    assert claims[1].driver_ids == () and claims[1].passage_ids == ("R1", "R2")  # repeated R1 counted once


def test_a_claim_the_packet_entails_is_a_patient_claim_and_needs_no_passage():
    c = counts(PATIENT, [R1])
    assert c["n_claims"] == 1 and c["n_external"] == 0
    assert rates(c)["citation_recall"] is None and rates(c)["pool_support"] is None


def test_a_correctly_cited_general_claim_is_fully_supported():
    c = counts(GENERAL, [R1, R2])
    assert (c["n_external"], c["n_cited_supported"], c["n_pool_supported"]) == (1, 1, 1)
    assert (c["n_citations"], c["n_citations_needed"]) == (1, 1)
    r = rates(c)
    assert (
        r["citation_recall"] == 1.0 and r["citation_precision"] == 1.0 and r["unsupported_claim_rate"] == 0.0
    )


def test_citing_the_wrong_passage_loses_recall_and_precision_but_not_pool_support():
    c = counts("Higher organ failure scores predict longer ICU stays [R2].", [R1, R2])
    assert (c["n_cited_supported"], c["n_pool_supported"], c["n_citations"], c["n_citations_needed"]) == (
        0,
        1,
        1,
        0,
    )
    r = rates(c)
    assert r["citation_recall"] == 0.0 and r["citation_precision"] == 0.0 and r["pool_support"] == 1.0


def test_a_redundant_second_citation_is_not_precise():
    c = counts("Higher organ failure scores predict longer ICU stays [R1] [R2].", [R1, R2])
    assert (c["n_cited_supported"], c["n_citations"], c["n_citations_needed"]) == (1, 2, 1)
    assert rates(c)["citation_precision"] == 0.5  # R1 alone entails it; R2 adds nothing


def test_two_passages_that_are_needed_together_are_both_precise():
    claim = (
        "Higher organ failure scores predict longer ICU stays "
        "and elevated creatinine marks recovery [R1] [R2]."
    )
    c = counts(claim, [R1, R2])
    assert (c["n_cited_supported"], c["n_citations"], c["n_citations_needed"]) == (1, 2, 2)


def test_an_uncited_claim_supported_by_the_pool_counts_for_pool_support_only():
    c = counts(
        "Higher organ failure scores predict longer ICU stays [D1].", [R1]
    )  # E6-like: nothing cited from R
    assert (c["n_external"], c["n_cited_supported"], c["n_pool_supported"], c["n_citations"]) == (1, 0, 1, 0)
    r = rates(c)
    assert r["citation_recall"] == 0.0 and r["citation_precision"] is None and r["pool_support"] == 1.0


def test_an_invented_claim_is_unsupported_by_the_whole_pool():
    c = counts("Vitamin supplements reverse organ failure completely [R1].", [R1, R2])
    assert c["n_external"] == 1 and c["n_pool_supported"] == 0
    assert rates(c)["unsupported_claim_rate"] == 1.0


def test_counts_add_up_over_several_explanations():
    a = counts(GENERAL, [R1, R2])
    b = counts("Vitamin supplements reverse organ failure completely [R1].", [R1, R2])
    total = {k: a[k] + b[k] for k in a}
    r = rates(total)
    assert r["citation_recall"] == 0.5 and r["pool_support"] == 0.5 and r["unsupported_claim_rate"] == 0.5


def test_kappa_on_worked_examples():
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)  # po 0.75, pe 0.5
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 0], [0, 1]) == -1.0
    assert math.isnan(cohens_kappa([1, 1], [1, 1]))  # chance agreement is 1
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


def test_checker_agreement_with_hand_labels():
    scores = "Higher organ failure scores predict longer stays."
    kidney = "Elevated creatinine marks kidney injury."
    pairs = [
        {"premise": scores, "claim": "Organ failure scores predict stays", "supported": True},
        {"premise": kidney, "claim": "Creatinine marks kidney injury", "supported": True},
        {"premise": kidney, "claim": "Vitamin supplements reverse injury", "supported": False},
        # the human says supported, the word-overlap checker says no:
        {"premise": kidney, "claim": "Kidney injury marks creatinine nonsense", "supported": True},
    ]
    out = checker_agreement(pairs, CHECK)
    assert out["n"] == 4 and out["accuracy"] == 0.75
    # human [1,1,0,1], machine [1,1,0,0]: po 0.75, pe = 0.75*0.5 + 0.25*0.5 = 0.5 -> kappa 0.5
    assert out["kappa"] == pytest.approx(0.5)


# ---- the Hugging Face checker, on a tiny random model built offline -------------------------------


@pytest.fixture(scope="module")
def tiny_nli(tmp_path_factory):
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification

    from fate_icu.llm.tiny import make_tiny_tokenizer

    path = tmp_path_factory.mktemp("nli")
    tokenizer = make_tiny_tokenizer([R1["text"], R2["text"], GENERAL, PATIENT])
    tokenizer.save_pretrained(path)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, pad_token_id=tokenizer.pad_token_id, num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )  # fmt: skip
    LlamaForSequenceClassification(config).save_pretrained(path)
    return path


def test_the_nli_checker_runs_on_a_tiny_model_and_is_deterministic(tiny_nli):
    checker = NLIChecker(str(tiny_nli), device="cpu", batch_size=2)
    pairs = [(R1["text"], GENERAL), (R2["text"], GENERAL), (R1["text"], PATIENT)]
    first = checker.entails_many(pairs)
    assert len(first) == 3 and all(isinstance(x, bool) for x in first)
    assert NLIChecker(str(tiny_nli), device="cpu").entails_many(pairs) == first
    assert checker.entails(*pairs[0]) == first[0]


def test_the_nli_checker_refuses_a_model_without_an_entailment_label(tiny_nli, tmp_path):
    import json
    import shutil

    folder = tmp_path / "bad"
    shutil.copytree(tiny_nli, folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config["id2label"] = {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="entailment label"):
        NLIChecker(str(folder), device="cpu").entails("a", "b")
