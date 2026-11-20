"""The explainer's prompt: fixed layout, worked examples that meet the contract."""

from __future__ import annotations

import copy

import pytest

from fate_icu.attribution.packet import EvidencePacket
from fate_icu.llm.prompt import EXAMPLES, INSTRUCTIONS, build_messages, render_evidence
from fate_icu.llm.validate import check


@pytest.mark.parametrize("index", range(len(EXAMPLES)))
def test_worked_examples_are_valid_packets_and_meet_the_contract(index):
    example = EXAMPLES[index]
    EvidencePacket.model_validate(example["packet"])
    verdict = check(example["text"], example["packet"], example["passages"])
    assert verdict.ok, verdict.failures


def test_render_is_deterministic_and_shows_every_driver_and_passage():
    example = EXAMPLES[0]
    a = render_evidence(example["packet"], example["passages"])
    b = render_evidence(copy.deepcopy(example["packet"]), copy.deepcopy(example["passages"]))
    assert a == b
    for d in example["packet"]["drivers"]:
        assert f"[{d['id']}]" in a and d["label"] in a
    assert "[R1]" in a and "PASSAGES" in a
    assert "points toward a longer stay" in a and "points toward a shorter stay" in a


def test_render_without_passages_has_no_passage_section():
    assert "PASSAGES" not in render_evidence(EXAMPLES[1]["packet"], [])


def test_render_accepts_a_pydantic_packet():
    packet = EvidencePacket.model_validate(EXAMPLES[0]["packet"])
    text = render_evidence(packet, EXAMPLES[0]["passages"])
    assert text == render_evidence(EXAMPLES[0]["packet"], EXAMPLES[0]["passages"])


def test_messages_are_one_user_turn_with_the_instructions():
    messages = build_messages(EXAMPLES[0]["packet"], EXAMPLES[0]["passages"])
    assert [m["role"] for m in messages] == ["user"]
    content = messages[0]["content"]
    assert content.startswith(INSTRUCTIONS) and content.endswith("Explanation:")


def test_few_shot_adds_both_examples_and_zero_shot_does_not():
    packet, passages = EXAMPLES[1]["packet"], []
    zero = build_messages(packet, passages)[0]["content"]
    few = build_messages(packet, passages, few_shot=True)[0]["content"]
    assert "Example 1" not in zero and "Example 2" in few and "Now the real case." in few
    assert len(few) > len(zero)
    for e in EXAMPLES:
        assert e["text"] in few and e["text"] not in zero


def test_instructions_state_the_contract_limits():
    assert "3 to 5 sentences" in INSTRUCTIONS
    assert "[D1]" in INSTRUCTIONS and "[R1]" in INSTRUCTIONS
    assert "treatment" in INSTRUCTIONS
