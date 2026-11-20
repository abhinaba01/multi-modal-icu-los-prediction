"""DPO after SFT: merge the adapter, then train a fresh adapter on preference pairs (tiny CPU model)."""

from __future__ import annotations

import pytest

from fate_icu.llm.dataset import build_records
from fate_icu.llm.dpo import DPOSettings, check_pairs, merge_adapter
from fate_icu.llm.dpo import train as dpo_train
from fate_icu.llm.prompt import build_messages
from fate_icu.llm.sft import SFTSettings
from fate_icu.llm.sft import train as sft_train
from fate_icu.llm.smoke import ScriptedSampler
from fate_icu.llm.synthetic import random_items
from fate_icu.llm.template import explain
from fate_icu.llm.tiny import make_tiny_model


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("dpo")
    items = random_items(6, seed=11)
    texts = [build_messages(p, ps)[0]["content"] for p, ps in items] + [explain(p) for p, _ in items]
    tiny = make_tiny_model(root / "tiny", texts)
    sft, dpo, _ = build_records(items, ScriptedSampler(items, seed=1), n=4, seed=1)
    settings = SFTSettings(
        run_name="sft", model=str(tiny), max_length=3072, max_steps=2, grad_accum=2, learning_rate=5e-3,
        gradient_checkpointing=False, save_steps=2, device="cpu",
    )  # fmt: skip
    sft_train(settings, sft, root / "sft")
    return root, tiny, dpo


def test_check_pairs_requires_the_three_fields_and_a_real_difference():
    good = {"prompt": [{"role": "user", "content": "q"}], "chosen": ["a"], "rejected": ["b"], "kind": "x"}
    assert check_pairs([good]) == [{k: good[k] for k in ("prompt", "chosen", "rejected")}]
    with pytest.raises(ValueError, match="lacks"):
        check_pairs([{"prompt": [], "chosen": []}])
    with pytest.raises(ValueError, match="identical"):
        check_pairs([{"prompt": [], "chosen": ["a"], "rejected": ["a"]}])
    with pytest.raises(ValueError, match="no preference pairs"):
        check_pairs([])


def test_merge_then_dpo_trains_a_fresh_adapter_and_resumes(prepared):
    root, tiny, pairs = prepared
    merged = merge_adapter(str(tiny), root / "sft" / "adapter", root / "merged")
    assert (merged / "config.json").exists() and (merged / "tokenizer_config.json").exists()
    settings = DPOSettings(
        run_name="dpo", max_length=3072, max_steps=2, grad_accum=2, learning_rate=1e-3, save_steps=2,
        gradient_checkpointing=False, device="cpu",
    )  # fmt: skip
    first = dpo_train(settings, str(merged), pairs[:8], root / "dpo")
    assert (
        first["steps"] == 2 and first["pairs"] == 8 and first["lora_modules"] == 14 and not first["resumed"]
    )
    assert (root / "dpo" / "adapter" / "adapter_config.json").exists()
    second = dpo_train(settings.model_copy(update={"max_steps": 3}), str(merged), pairs[:8], root / "dpo")
    assert second["resumed"] is True and second["steps"] == 3
