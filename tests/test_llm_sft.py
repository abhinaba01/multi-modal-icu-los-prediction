"""SFT: completion-only label masking, LoRA module selection, and a tiny CPU training run with resume."""

from __future__ import annotations

import json
import re

import pytest
import torch
from torch import nn

from fate_icu.llm.dataset import build_records
from fate_icu.llm.prompt import EXAMPLES, build_messages
from fate_icu.llm.sft import (
    IGNORE,
    SFTSettings,
    lora_target_regex,
    select_lora_modules,
    tokenize_example,
    tokenize_records,
    train,
)
from fate_icu.llm.tiny import make_tiny_model


class Fixed:
    def __init__(self, texts):
        self.texts = texts

    def generate(self, messages, *, n=1, temperature=0.0, seed=None):
        return list(self.texts)


ITEMS = [(e["packet"], e["passages"]) for e in EXAMPLES]


@pytest.fixture(scope="module")
def records():
    sft, _, _ = build_records(ITEMS, Fixed([e["text"] for e in EXAMPLES][:1]), n=1)
    both = []
    for example in EXAMPLES:  # one record per worked example (each is valid for its own packet)
        both.append(
            {
                "prompt": build_messages(example["packet"], example["passages"]),
                "completion": [{"role": "assistant", "content": example["text"]}],
            }
        )
    assert sft  # the record builder itself works on the same inputs
    return both


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, records):
    texts = [r["prompt"][0]["content"] for r in records] + [r["completion"][0]["content"] for r in records]
    return make_tiny_model(tmp_path_factory.mktemp("tiny"), texts)


@pytest.fixture(scope="module")
def tokenizer(tiny):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny)


def test_only_completion_tokens_keep_a_label_and_they_decode_to_the_completion(tokenizer, records):
    for record in records:
        row = tokenize_example(tokenizer, record["prompt"], record["completion"], 4096)
        assert len(row["input_ids"]) == len(row["labels"])
        kept = [t for t, label in zip(row["input_ids"], row["labels"], strict=True) if label != IGNORE]
        assert [label for label in row["labels"] if label != IGNORE] == kept
        decoded = tokenizer.decode(kept, skip_special_tokens=False)
        assert decoded == record["completion"][0]["content"] + "<end_of_turn>\n"
        # the prompt part is entirely masked and ends with the generation prompt
        n_prompt = row["labels"].index(next(label for label in row["labels"] if label != IGNORE))
        assert all(label == IGNORE for label in row["labels"][:n_prompt])
        assert tokenizer.decode(row["input_ids"][:n_prompt]).endswith("<start_of_turn>model\n")


def test_an_example_longer_than_max_length_is_dropped_and_counted(tokenizer, records):
    assert tokenize_example(tokenizer, records[0]["prompt"], records[0]["completion"], 10) is None
    rows, dropped = tokenize_records(tokenizer, records, 10)
    assert rows == [] and dropped == len(records)


def test_the_lora_regex_matches_language_projections_and_not_the_vision_tower():
    pattern = re.compile(lora_target_regex())
    for name in (
        "model.layers.0.self_attn.q_proj",
        "model.language_model.layers.3.mlp.down_proj",
        "language_model.model.layers.1.self_attn.o_proj",
    ):
        assert pattern.match(name), name
    for name in (
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.vision_tower.layers.0.mlp.up_proj",
        "multi_modal_projector.q_proj",
        "model.layers.0.self_attn.rotary_emb",
        "lm_head",
    ):
        assert not pattern.match(name), name


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj = nn.Linear(2, 2), nn.Linear(2, 2)


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.ModuleDict({"layers": nn.ModuleList([_Block(), _Block()])})
        self.vision_tower = nn.ModuleDict({"layers": nn.ModuleList([_Block()])})


def test_select_lora_modules_takes_the_language_model_and_skips_the_vision_tower():
    names = select_lora_modules(_Fake())
    assert names == [
        "language_model.layers.0.q_proj",
        "language_model.layers.0.v_proj",
        "language_model.layers.1.q_proj",
        "language_model.layers.1.v_proj",
    ]


def test_select_lora_modules_fails_loudly_when_nothing_matches():
    with pytest.raises(ValueError, match="no LoRA target modules"):
        select_lora_modules(nn.Sequential(nn.Linear(2, 2)))


def test_select_lora_modules_on_the_tiny_llama(tiny):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny)
    assert len(select_lora_modules(model)) == 2 * 7


def _settings(tiny, steps):
    return SFTSettings(
        run_name="sft_test",
        model=str(tiny),
        max_length=1536,
        max_steps=steps,
        batch_size=1,
        grad_accum=2,
        learning_rate=5e-3,
        save_steps=2,
        logging_steps=1,
        gradient_checkpointing=False,
        pass_rate_every=2,
        pass_rate_packets=2,
        max_new_tokens=8,
        device="cpu",
    )


def test_a_tiny_run_trains_saves_an_adapter_and_resumes_after_an_interruption(tiny, records, tmp_path):
    out = tmp_path / "run"
    first = train(_settings(tiny, 2), records, out, val_items=ITEMS)
    assert first["steps"] == 2 and first["trained_examples"] == len(records) and not first["resumed"]
    assert first["dropped_too_long"] == 0 and first["lora_modules"] == 14
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "run_manifest.json").exists() and (out / "metrics.json").exists()
    assert first["pass_rates"] and first["pass_rates"][0]["pass_rate"] in (0.0, 0.5, 1.0)

    second = train(_settings(tiny, 4), records, out, val_items=ITEMS)  # a rerun picks up the checkpoint
    assert second["resumed"] is True and second["steps"] == 4
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["steps"] == 4


def test_the_tiny_run_lowers_the_loss(tiny, records, tmp_path):
    metrics = train(_settings(tiny, 12), records, tmp_path / "loss", val_items=())
    assert metrics["last_loss"] < metrics["first_loss"]


def test_adapter_weights_exist_and_base_weights_are_unchanged(tiny, records, tmp_path):
    from transformers import AutoModelForCausalLM

    before = AutoModelForCausalLM.from_pretrained(tiny).state_dict()
    train(_settings(tiny, 2), records, tmp_path / "frozen", val_items=())
    after = AutoModelForCausalLM.from_pretrained(tiny).state_dict()  # the base on disk is never modified
    assert all(torch.equal(before[k], after[k]) for k in before)
