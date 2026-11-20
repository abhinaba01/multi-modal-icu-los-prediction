"""Synthetic packets and the Stage 6 smoke run."""

from __future__ import annotations

import json

from fate_icu.attribution.packet import EvidencePacket
from fate_icu.llm.smoke import run
from fate_icu.llm.synthetic import random_items
from fate_icu.llm.template import explain
from fate_icu.llm.validate import check


def test_synthetic_packets_are_valid_and_the_template_always_meets_the_contract():
    items = random_items(200, seed=3)
    for packet, passages in items:
        EvidencePacket.model_validate(packet)
        assert 2 <= len(packet["drivers"]) <= 5
        assert check(explain(packet), packet, passages).ok


def test_synthetic_items_are_reproducible_and_vary():
    a, b, c = random_items(20, seed=1), random_items(20, seed=1), random_items(20, seed=2)
    assert a == b and a != c
    assert any(passages for _, passages in a) and any(not passages for _, passages in a)


def test_the_smoke_run_goes_end_to_end_and_reports_counts_only(tmp_path):
    summary = run(tmp_path, n_train=8, n_val=2, seed=0)
    assert summary["dataset"]["sft_records"] > 0 and summary["dataset"]["dpo_pairs"]
    assert summary["sft"]["steps"] == 6 and summary["sft"]["lora_modules"] == 14
    assert summary["generation"]["n"] == 2
    for name in ("sft.jsonl", "dpo.jsonl", "smoke_summary.json"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "sft" / "adapter" / "adapter_config.json").exists()
    assert json.loads((tmp_path / "smoke_summary.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(summary, default=float)
    )


def test_the_command_line_steps_sample_and_generate_with_resume(tmp_path, capsys):
    from fate_icu.llm.dataset import write_jsonl
    from fate_icu.llm.prompt import build_messages
    from fate_icu.llm.run import load_items, main
    from fate_icu.llm.tiny import make_tiny_model

    items = random_items(4, seed=7)
    path = tmp_path / "items.jsonl"
    write_jsonl(path, [{"packet": p, "passages": ps} for p, ps in items])
    assert len(load_items(path)) == 4
    texts = [build_messages(p, ps)[0]["content"] for p, ps in items] + [explain(p) for p, _ in items]
    tiny = make_tiny_model(tmp_path / "tiny", texts)

    main(
        [
            "sample",
            "--items",
            str(path),
            "--model",
            str(tiny),
            "--out",
            str(tmp_path / "s"),
            "--samples",
            "2",
            "--max-new-tokens",
            "6",
            "--device",
            "cpu",
        ]
    )
    stats = json.loads((tmp_path / "s" / "dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["prompts"] == 4 and stats["samples"] == 8  # a random model proposes nothing valid

    out = tmp_path / "g"
    args = [
        "generate",
        "--items",
        str(path),
        "--model",
        str(tiny),
        "--out",
        str(out),
        "--max-new-tokens",
        "6",
        "--device",
        "cpu",
    ]
    main(args)
    lines = (out / "explanations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and all(json.loads(x)["source"] == "template" for x in lines)  # always valid
    # an interrupted run: drop the last two lines, resume, and the file is whole again
    (out / "explanations.jsonl").write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    main([*args, "--resume"])
    again = (out / "explanations.jsonl").read_text(encoding="utf-8").splitlines()
    assert again == lines
    assert json.loads((out / "generation_stats.json").read_text(encoding="utf-8"))["resumed_from"] == 2


def test_load_items_rejects_a_line_without_a_packet(tmp_path):
    import pytest

    from fate_icu.llm.dataset import write_jsonl
    from fate_icu.llm.run import load_items

    write_jsonl(tmp_path / "bad.jsonl", [{"passages": []}])
    with pytest.raises(ValueError, match="no 'packet'"):
        load_items(tmp_path / "bad.jsonl")
