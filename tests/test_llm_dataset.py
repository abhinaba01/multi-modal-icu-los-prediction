"""Training data: rejection sampling, corrupted negatives, SFT records and DPO pairs."""

from __future__ import annotations

import random

import pytest

from fate_icu.llm.dataset import (
    CORRUPTIONS,
    build_records,
    corrupt,
    corrupted_negatives,
    read_jsonl,
    sample_candidates,
    write_jsonl,
)
from fate_icu.llm.prompt import EXAMPLES
from fate_icu.llm.validate import check

PACKET, PASSAGES, GOOD = EXAMPLES[0]["packet"], EXAMPLES[0]["passages"], EXAMPLES[0]["text"]
PACKET2, GOOD2 = EXAMPLES[1]["packet"], EXAMPLES[1]["text"]
BAD = GOOD.replace("7.4", "9.9")


class Mixed:
    """Returns the same list of texts for every call (n is ignored: the list is the n samples)."""

    def __init__(self, texts):
        self.texts, self.calls = texts, 0

    def generate(self, messages, *, n=1, temperature=0.0, seed=None):
        self.calls += 1
        return list(self.texts)


def test_sample_candidates_marks_each_sample_against_the_contract():
    got = sample_candidates(Mixed([GOOD, BAD, "nonsense"]), PACKET, PASSAGES, n=3)
    assert [c.ok for c in got] == [True, False, False]
    assert "number_not_in_packet" in got[1].failures


def test_an_extra_checker_can_reject_an_otherwise_valid_sample():
    got = sample_candidates(Mixed([GOOD]), PACKET, PASSAGES, extra_check=lambda *a: False)
    assert got[0].ok is False and got[0].failures == ("support_check",)


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_each_corruption_changes_the_text_and_breaks_the_contract(kind):
    assert check(GOOD, PACKET, PASSAGES).ok
    bad = corrupt(GOOD, PACKET, kind, random.Random(0))
    assert bad is not None and bad != GOOD
    assert not check(bad, PACKET, PASSAGES).ok


def test_flip_direction_swaps_the_direction_word_and_names_the_rule_broken():
    bad = corrupt(GOOD, PACKET, "flip_direction", random.Random(1))
    assert "direction" in check(bad, PACKET, PASSAGES).failures


def test_miscite_uses_an_unknown_id():
    bad = corrupt(GOOD, PACKET, "miscite", random.Random(2))
    assert "unknown_id" in check(bad, PACKET, PASSAGES).failures


def test_a_text_without_passages_is_miscited_with_an_unknown_driver():
    bad = corrupt(GOOD2, PACKET2, "miscite", random.Random(3))
    assert "[D9]" in bad and "unknown_id" in check(bad, PACKET2, []).failures


def test_corrupted_negatives_only_keep_texts_that_really_fail():
    for text, packet, passages in ((GOOD, PACKET, PASSAGES), (GOOD2, PACKET2, [])):
        out = corrupted_negatives(text, packet, passages, random.Random(0))
        assert out and all(not check(bad, packet, passages).ok for _, bad in out)


def test_unknown_corruption_kind_is_an_error():
    with pytest.raises(ValueError, match="unknown corruption"):
        corrupt(GOOD, PACKET, "nope", random.Random(0))


def test_build_records_makes_sft_records_and_pairs_with_a_zero_shot_prompt():
    backend = Mixed([GOOD, BAD, GOOD, "nonsense"])
    sft, dpo, stats = build_records([(PACKET, PASSAGES)], backend, n=4, seed=5)
    assert len(sft) == 1  # the duplicate good sample is dropped
    record = sft[0]
    assert record["completion"] == [{"role": "assistant", "content": GOOD}]
    assert record["prompt"][0]["role"] == "user" and "Example 1" not in record["prompt"][0]["content"]
    kinds = {p["kind"] for p in dpo}
    assert "failed_sample" in kinds and kinds & set(CORRUPTIONS)
    for pair in dpo:
        assert pair["chosen"][0]["content"] == GOOD
        assert not check(pair["rejected"][0]["content"], PACKET, PASSAGES).ok
    summary = stats.as_dict()
    assert summary["prompts"] == 1 and summary["samples"] == 4
    assert summary["sample_pass_rate"] == pytest.approx(0.5)
    assert summary["sft_records"] == 1 and sum(summary["dpo_pairs"].values()) == len(dpo)


def test_prompts_with_no_valid_sample_are_counted_and_make_no_record():
    sft, dpo, stats = build_records([(PACKET, PASSAGES)], Mixed(["x", BAD]), n=2)
    assert sft == [] and dpo == [] and stats.prompts_without_a_kept_sample == 1


def test_max_kept_limits_the_records_per_prompt():
    other = GOOD.replace("Age 41", "An age of 41")
    assert check(other, PACKET, PASSAGES).ok
    sft, _, _ = build_records([(PACKET, PASSAGES)], Mixed([GOOD, other]), n=2, max_kept=1)
    assert len(sft) == 1


def test_building_records_is_deterministic():
    items = [(PACKET, PASSAGES), (PACKET2, [])]
    a = build_records(items, Mixed([GOOD, BAD]), n=2, seed=9)
    b = build_records(items, Mixed([GOOD, BAD]), n=2, seed=9)
    assert a[0] == b[0] and a[1] == b[1]


def test_jsonl_round_trip(tmp_path):
    records = [{"a": 1, "b": "é"}, {"a": 2, "b": "x"}]
    assert write_jsonl(tmp_path / "sub" / "r.jsonl", records) == 2
    assert read_jsonl(tmp_path / "sub" / "r.jsonl") == records
