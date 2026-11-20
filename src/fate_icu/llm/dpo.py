"""Optional DPO after SFT (Stage 6; hypothesis H5).

    python -m fate_icu.llm.dpo merge --model <base> --adapter <sft_dir>/adapter --out <merged_dir>
    python -m fate_icu.llm.dpo train --config configs/llm/dpo_full.yaml --model <merged_dir> \
        --pairs <dpo.jsonl> --out <dpo_dir>

Recipe: merge the SFT adapter into a copy of the base model (on the CPU, in bf16 or float32) so that the
*reference policy is the SFT model*; then train a fresh LoRA adapter on that merged model with TRL's
``DPOTrainer`` on the preference pairs of ``llm/dataset.py`` (chosen: passes every check; rejected: a failed
sample or a corrupted copy). Reference log-probabilities are precomputed once, beta is 0.1, the learning rate
is low (5e-6) and one epoch is enough. Checkpoints are saved so a reclaimed spot VM resumes.

Imports of torch, peft, trl and transformers are inside the functions: the module loads without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fate_icu.llm.dataset import read_jsonl
from fate_icu.llm.sft import lora_target_regex, select_lora_modules


class DPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    revision: str | None = None
    image_text_model: bool = False
    quantize_4bit: bool = False
    beta: float = Field(0.1, gt=0)
    learning_rate: float = 5e-6
    epochs: float = Field(1.0, gt=0)
    max_steps: int | None = None
    max_length: int = Field(2048, gt=0)
    batch_size: int = Field(1, gt=0)
    grad_accum: int = Field(16, gt=0)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    optim: str = "adamw_torch"
    bf16: bool = False
    gradient_checkpointing: bool = True
    precompute_ref_log_probs: bool = True
    save_steps: int = 50
    logging_steps: int = 5
    seed: int = 666
    device: str = "auto"


def load_settings(path: str | Path) -> DPOSettings:
    return DPOSettings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def merge_adapter(
    base: str,
    adapter: str | Path,
    out_dir: str | Path,
    *,
    revision: str | None = None,
    image_text_model: bool = False,
) -> Path:
    """Merge a LoRA adapter into a copy of ``base`` on the CPU; save model and tokenizer to ``out_dir``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    auto = AutoModelForImageTextToText if image_text_model else AutoModelForCausalLM
    model = auto.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(out)
    return out


def check_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pairs in the form TRL expects (prompt, chosen, rejected); fails loudly on a malformed one."""
    clean = []
    for number, pair in enumerate(pairs, start=1):
        missing = {"prompt", "chosen", "rejected"} - pair.keys()
        if missing:
            raise ValueError(f"pair {number} lacks {sorted(missing)}")
        if pair["chosen"] == pair["rejected"]:
            raise ValueError(f"pair {number}: chosen and rejected are identical")
        clean.append({k: pair[k] for k in ("prompt", "chosen", "rejected")})
    if not clean:
        raise ValueError("no preference pairs to train on")
    return clean


def train(
    settings: DPOSettings, model_path: str, pairs: list[dict[str, Any]], out_dir: str | Path
) -> dict[str, Any]:
    """DPO-train a fresh LoRA adapter on ``model_path`` (the merged SFT model); resumes from a checkpoint."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from fate_icu.utils.device import get_device
    from fate_icu.utils.manifest import write_run_manifest
    from fate_icu.utils.seed import seed_everything

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)
    device = get_device(settings.device)
    rows = check_pairs(pairs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    auto = AutoModelForImageTextToText if settings.image_text_model else AutoModelForCausalLM
    dtype = torch.bfloat16 if settings.bf16 else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype}
    if settings.quantize_4bit:
        from transformers import BitsAndBytesConfig  # needs bitsandbytes (the gpu extra)

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    model = auto.from_pretrained(model_path, **kwargs)
    if settings.quantize_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=settings.gradient_checkpointing
        )
    else:
        model = model.to(device)
    selected = select_lora_modules(model)
    print(f"LoRA on {len(selected)} language-model modules")
    lora = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=lora_target_regex(),
        task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(out),
        beta=settings.beta,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps if settings.max_steps is not None else -1,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        max_length=settings.max_length,
        optim=settings.optim,
        bf16=settings.bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if settings.gradient_checkpointing else None,
        precompute_ref_log_probs=settings.precompute_ref_log_probs,
        logging_steps=settings.logging_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=settings.seed,
        use_cpu=device == "cpu",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora,
    )
    resume = any(out.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.model.save_pretrained(out / "adapter")
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "pairs": len(rows),
        "lora_modules": len(selected),
        "steps": trainer.state.global_step,
        "resumed": resume,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_loss": result.training_loss,
        "device": device,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_manifest(
        out,
        run_id=settings.run_name,
        config=settings.model_dump(),
        seeds=[settings.seed],
        model_revisions={model_path: settings.revision or "local"},
        device=device,
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--model", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--revision")
    merge.add_argument("--image-text-model", action="store_true")
    fit = sub.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--model", required=True, help="the merged SFT model folder")
    fit.add_argument("--pairs", required=True)
    fit.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        path = merge_adapter(
            args.model, args.adapter, args.out, revision=args.revision, image_text_model=args.image_text_model
        )
        print(json.dumps({"merged_model": str(path)}))
    else:
        metrics = train(load_settings(args.config), args.model, read_jsonl(args.pairs), args.out)
        print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
