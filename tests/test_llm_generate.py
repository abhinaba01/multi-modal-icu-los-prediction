"""Generating explanations: retry, fallback, counts, and the Hugging Face backend on a tiny random model."""

from __future__ import annotations

import pytest

from fate_icu.llm.generate import HFBackend, RunStats, clean, explain_many, explain_one
from fate_icu.llm.prompt import EXAMPLES
from fate_icu.llm.validate import check

PACKET, PASSAGES = EXAMPLES[0]["packet"], EXAMPLES[0]["passages"]
GOOD = EXAMPLES[0]["text"]
BAD_NUMBER = GOOD.replace("7.4", "9.9")  # a made-up prediction


class Scripted:
    """Returns the scripted texts in order and remembers how it was called."""

    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def generate(self, messages, *, n=1, temperature=0.0, seed=None):
        self.calls.append({"temperature": temperature, "seed": seed, "n": n, "messages": messages})
        return [self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]] * n


def test_a_valid_first_answer_is_used_as_is_with_one_greedy_call():
    backend = Scripted([GOOD])
    result = explain_one(backend, PACKET, PASSAGES)
    assert (result.text, result.source, result.attempts, result.failures) == (GOOD, "model", 1, ())
    assert len(backend.calls) == 1 and backend.calls[0]["temperature"] == 0.0


def test_a_failed_first_answer_is_resampled_at_a_higher_temperature():
    backend = Scripted([BAD_NUMBER, GOOD])
    result = explain_one(backend, PACKET, PASSAGES, seed=10)
    assert result.source == "model" and result.attempts == 2
    assert result.failures and "number_not_in_packet" in result.failures[0]
    assert [c["temperature"] for c in backend.calls] == [0.0, 0.7]
    assert [c["seed"] for c in backend.calls] == [10, 11]


def test_after_three_failed_resamples_the_template_is_used_and_it_meets_the_contract():
    backend = Scripted([BAD_NUMBER])
    result = explain_one(backend, PACKET, PASSAGES)
    assert result.source == "template" and result.attempts == 4 and len(result.failures) == 4
    assert len(backend.calls) == 4
    assert check(result.text, PACKET, PASSAGES).ok


def test_the_returned_text_always_meets_the_contract():
    for texts in ([GOOD], [BAD_NUMBER], ["nonsense"], ["", GOOD]):
        result = explain_one(Scripted(texts), PACKET, PASSAGES)
        assert check(result.text, PACKET, PASSAGES).ok, texts


def test_the_prompt_sent_to_the_backend_holds_the_packet_and_few_shot_is_optional():
    zero, few = Scripted([GOOD]), Scripted([GOOD])
    explain_one(zero, PACKET, PASSAGES)
    explain_one(few, PACKET, PASSAGES, few_shot=True)
    assert "Example 1" not in zero.calls[0]["messages"][0]["content"]
    assert "Example 1" in few.calls[0]["messages"][0]["content"]


def test_clean_removes_the_label_and_whitespace():
    assert clean("  Explanation: Hello there.  ") == "Hello there."
    assert clean("explanation:\nHello") == "Hello"
    assert clean("No label.") == "No label."


def test_stats_count_retries_fallbacks_and_broken_rules():
    items = [(PACKET, PASSAGES)] * 3
    backends = iter([Scripted([GOOD]), Scripted([BAD_NUMBER, GOOD]), Scripted([BAD_NUMBER])])

    class Routing:  # one scripted backend per packet
        def generate(self, messages, **kw):
            return self.current.generate(messages, **kw)

    router = Routing()
    stats = RunStats()
    for packet, passages in items:
        router.current = next(backends)
        stats.add(explain_one(router, packet, passages))
    rates = stats.rates()
    assert rates["n"] == 3
    assert rates["retry_rate"] == pytest.approx(2 / 3)
    assert rates["fallback_rate"] == pytest.approx(1 / 3)
    assert rates["rule_failures"]["number_not_in_packet"] == 5  # 1 + 4 failed attempts


def test_explain_many_gives_each_packet_its_own_seeds_and_counts():
    backend = Scripted([GOOD])
    results, stats = explain_many(backend, [(PACKET, PASSAGES)] * 3, seed=100)
    assert len(results) == 3 and stats.rates()["fallback_rate"] == 0
    assert [c["seed"] for c in backend.calls] == [100, 104, 108]


def test_hf_backend_imports_lazily_and_loads_nothing_until_used():
    backend = HFBackend("not-a-real-model")
    assert backend._model is None and backend._tokenizer is None
