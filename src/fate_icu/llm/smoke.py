"""The Stage 6 smoke run: every part of the explainer on a laptop CPU, in about a minute, no download.

    uv run --no-sync python -m fate_icu.llm.smoke --out artifacts/llm_smoke

It uses synthetic packets (``synthetic.py``) and a tiny random model built on the spot (``tiny.py``). The
"sampler" that proposes explanations is a scripted stand-in (the deterministic template plus a few corrupted
copies), because a random model proposes nothing valid; it exists to feed the rest of the pipeline:

    synthetic packets -> sampling -> contract check -> SFT records + DPO pairs
    -> SFT with LoRA (with a mid-run pass-rate log) -> generation with the adapter -> retry/fallback counts

Nothing here says anything about explanation quality; it shows that the code runs, saves and resumes. The
numbers it prints are plumbing counts, never results (CLAUDE.md A5).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from fate_icu.llm.dataset import CORRUPTIONS, build_records, corrupt, write_jsonl
from fate_icu.llm.generate import HFBackend, explain_many
from fate_icu.llm.prompt import build_messages
from fate_icu.llm.sft import SFTSettings, load_settings, train
from fate_icu.llm.synthetic import random_items
from fate_icu.llm.template import explain as template_explain
from fate_icu.llm.tiny import make_tiny_model

CONFIG = Path("configs/llm/smoke.yaml")


class ScriptedSampler:
    """Proposes the template explanation of the packet in the prompt, some copies corrupted.

    It reads the packet back from a lookup keyed by the prompt text, so it can answer the prompts the dataset
    builder sends (which carry the packet only as rendered text).
    """

    def __init__(self, items: list[tuple[Any, Any]], seed: int = 0):
        self.rng = random.Random(seed)
        self.by_prompt = {build_messages(p, ps, few_shot=True)[0]["content"]: p for p, ps in items}

    def generate(
        self, messages: list[dict[str, str]], *, n: int = 1, temperature: float = 0.0, seed: int | None = None
    ):
        packet = self.by_prompt[messages[0]["content"]]
        good = template_explain(packet)
        out = []
        for i in range(n):
            if i % 2 == 0:
                out.append(good)
            else:
                bad = corrupt(good, packet, CORRUPTIONS[(i // 2) % len(CORRUPTIONS)], self.rng)
                out.append(bad or good)
        return out


def run(out_dir: str | Path, *, n_train: int = 24, n_val: int = 6, seed: int = 0) -> dict[str, Any]:
    out = Path(out_dir)
    items = random_items(n_train + n_val, seed=seed)
    train_items, val_items = items[:n_train], items[n_train:]
    texts = [build_messages(p, ps)[0]["content"] for p, ps in items] + [template_explain(p) for p, _ in items]
    tiny = make_tiny_model(out / "tiny", texts)

    sft, dpo, stats = build_records(train_items, ScriptedSampler(train_items, seed), n=4, seed=seed)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "dpo.jsonl", dpo)

    settings = load_settings(CONFIG).model_copy(update={"model": str(tiny)})
    metrics = train(settings, sft, out / "sft", val_items=val_items)

    backend = HFBackend(str(tiny), adapter=str(out / "sft" / "adapter"), max_new_tokens=60, device="cpu")
    _, gen_stats = explain_many(backend, val_items, seed=seed)
    summary = {
        "dataset": stats.as_dict(),
        "sft": {
            k: metrics[k]
            for k in ("trained_examples", "dropped_too_long", "lora_modules", "steps", "last_loss")
        },
        "generation": gen_stats.rates(),
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/llm_smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.out, seed=args.seed), indent=2, default=float))


if __name__ == "__main__":
    main()


__all__ = ["SFTSettings", "ScriptedSampler", "main", "run"]
