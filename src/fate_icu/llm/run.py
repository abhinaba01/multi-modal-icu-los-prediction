"""Command-line steps of the explainer for the GCP VM (Stage 6).

    python -m fate_icu.llm.run sample   --items items.jsonl --model <id or folder> --out <dir>
    python -m fate_icu.llm.run generate --items items.jsonl --model <hf id or folder> --adapter <dir> \
        --out <dir>

``items.jsonl`` has one JSON object per line:
``{"packet": {...}, "passages": [{"rid": "R1", "text": ...}, ...]}`` (the Stage 4 packet and the Stage 5
retrieved passages of one stay; ``passages_for_prompt`` gives the shape). The files are credentialed when
the packets are: keep them under the VM's artifacts folder.

* ``sample`` samples the base model (few-shot, temperature 0.8) for every item, keeps the explanations that
  meet the contract and writes ``sft.jsonl``, ``dpo.jsonl`` and ``dataset_stats.json`` (counts only).
* ``generate`` writes an explanation for every item (greedy; resample up to 3 times; template fallback) to
  ``explanations.jsonl`` and the retry and fallback rates to ``generation_stats.json``. A spot VM can be
  reclaimed, so finished items are saved as it goes and ``--resume`` skips them.

Printed output is aggregates only: counts and rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_icu.llm.dataset import build_records, read_jsonl, write_jsonl
from fate_icu.llm.generate import HFBackend, RunStats, explain_one


def load_items(path: str | Path) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    items = []
    for number, record in enumerate(read_jsonl(path), start=1):
        if "packet" not in record:
            raise ValueError(f"{path} line {number} has no 'packet'")
        items.append((record["packet"], record.get("passages", [])))
    if not items:
        raise ValueError(f"{path} holds no items")
    return items


def _backend(args: argparse.Namespace) -> HFBackend:
    return HFBackend(
        args.model,
        revision=args.revision,
        adapter=getattr(args, "adapter", None),
        max_new_tokens=args.max_new_tokens,
        image_text_model=args.image_text_model,
        device=args.device,
    )


def sample(args: argparse.Namespace) -> dict[str, Any]:
    items = load_items(args.items)
    sft, dpo, stats = build_records(
        items, _backend(args), n=args.samples, temperature=args.temperature, seed=args.seed
    )
    out = Path(args.out)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "dpo.jsonl", dpo)
    summary = stats.as_dict()
    (out / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate(args: argparse.Namespace) -> dict[str, Any]:
    items = load_items(args.items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "explanations.jsonl"
    done = len(read_jsonl(target)) if args.resume and target.exists() else 0
    if not args.resume and target.exists():
        target.unlink()
    backend, stats = _backend(args), RunStats()
    block = 4  # seeds per item: the first try and three resamples
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for index, (packet, passages) in enumerate(items):
            if index < done:
                continue
            result = explain_one(backend, packet, passages, seed=args.seed + index * block)
            stats.add(result)
            record = {
                "index": index,
                "text": result.text,
                "source": result.source,
                "attempts": result.attempts,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    summary = {"items": len(items), "resumed_from": done, **stats.rates()}
    (out / "generation_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("sample", sample), ("generate", generate)):
        p = sub.add_parser(name)
        p.add_argument("--items", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--revision")
        p.add_argument("--out", required=True)
        p.add_argument("--max-new-tokens", type=int, default=300)
        p.add_argument(
            "--image-text-model", action="store_true", help="MedGemma loads as an image-text model"
        )
        p.add_argument("--device")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(handler=handler)
        if name == "sample":
            p.add_argument("--samples", type=int, default=8)
            p.add_argument("--temperature", type=float, default=0.8)
        else:
            p.add_argument("--adapter")
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(args.handler(args), indent=2, default=float))


if __name__ == "__main__":
    main()
