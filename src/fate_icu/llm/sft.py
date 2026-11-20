"""QLoRA supervised fine-tuning of the explainer (Stage 6).

    python -m fate_icu.llm.sft --config configs/llm/smoke.yaml --records <sft.jsonl> --out <dir>

The records are prompt-completion pairs from ``llm/dataset.py``. Each is tokenised here with the model's chat
template and the loss is taken on the completion only: ``labels`` is -100 on every prompt token (tested: the
tokens that keep a label decode to exactly the completion plus the end-of-turn marker). TRL's ``SFTTrainer``
receives the tokenised dataset, so the loss, optimiser, scheduler and checkpoints are its own.

* **LoRA** goes on the language model's attention and MLP projections only. MedGemma also has a vision tower
  with the same module names, so the modules are chosen by a regular expression that excludes it, and the
  count is printed and checked (``select_lora_modules``).
* **QLoRA** (``quantize_4bit: true``): 4-bit NF4, double quantisation, bf16 compute, through ``bitsandbytes``,
  which is imported lazily and exists only on the GCP VM (the ``gpu`` extra).
* **Spot VMs:** checkpoints are saved every ``save_steps`` and a rerun resumes from the newest one.
* **Pass rate:** every ``pass_rate_every`` steps the model writes explanations for the validation packets and
  the share that meets the contract is logged (an early sign of whether training works).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fate_icu.llm.dataset import read_jsonl
from fate_icu.llm.generate import LoadedBackend
from fate_icu.llm.prompt import build_messages
from fate_icu.llm.validate import check

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NOT_LANGUAGE = ("vision", "visual", "multi_modal_projector", "mm_projector", "image")
IGNORE = -100


class SFTSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    model: str
    revision: str | None = None
    image_text_model: bool = False  # MedGemma loads with AutoModelForImageTextToText
    quantize_4bit: bool = False
    max_length: int = Field(2048, gt=0)
    epochs: float = Field(2.0, gt=0)
    max_steps: int | None = None  # overrides epochs (smoke runs)
    batch_size: int = Field(1, gt=0)
    grad_accum: int = Field(16, gt=0)
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    optim: str = "adamw_torch"  # the full config uses paged_adamw_8bit
    bf16: bool = False
    gradient_checkpointing: bool = True
    save_steps: int = 50
    logging_steps: int = 5
    pass_rate_every: int = 0  # 0: never
    pass_rate_packets: int = 100
    max_new_tokens: int = 300
    seed: int = 666
    device: str = "auto"


def load_settings(path: str | Path) -> SFTSettings:
    return SFTSettings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---- LoRA modules --------------------------------------------------------------------------------------


def lora_target_regex() -> str:
    """Every attention/MLP projection whose path names no vision or projector part."""
    excluded = "|".join(re.escape(x) for x in NOT_LANGUAGE)
    return rf"^(?!.*(?:{excluded})).*\.(?:{'|'.join(PROJECTIONS)})$"


def select_lora_modules(model: Any) -> list[str]:
    """The names of the modules ``lora_target_regex`` selects, checked to be a language-model-only set."""
    pattern = re.compile(lora_target_regex())
    names = [n for n, _ in model.named_modules() if pattern.match(n)]
    if not names:
        raise ValueError(
            "no LoRA target modules found: the model's projection names are not the expected ones"
        )
    bad = [n for n in names if any(x in n for x in NOT_LANGUAGE)]
    if bad:
        raise ValueError(f"LoRA would touch a non-language module: {bad[0]}")
    return names


# ---- tokenisation with a completion-only loss ---------------------------------------------------------


def _ids(tokenizer: Any, messages: list[dict[str, str]], *, generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=True)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def tokenize_example(
    tokenizer: Any, prompt: list[dict[str, str]], completion: list[dict[str, str]], max_length: int
) -> dict[str, list[int]] | None:
    """``input_ids`` and ``labels`` for one record (labels are -100 on the prompt), or ``None`` if too long.

    The prompt is rendered with the generation prompt (``<start_of_turn>model``), the whole chat without it;
    the completion is whatever the whole chat adds, so it includes the end-of-turn marker the model must learn
    to emit. The prompt tokens must be a prefix of the whole chat, otherwise the template is not usable here.
    """
    prompt_ids = _ids(tokenizer, prompt, generation_prompt=True)
    full_ids = _ids(tokenizer, prompt + completion, generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("the chat template does not render the prompt as a prefix of the whole chat")
    if len(full_ids) > max_length:
        return None
    labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


def tokenize_records(
    tokenizer: Any, records: Sequence[dict[str, Any]], max_length: int
) -> tuple[list[dict[str, list[int]]], int]:
    """The tokenised examples and how many records were dropped for exceeding ``max_length``."""
    rows, dropped = [], 0
    for record in records:
        row = tokenize_example(tokenizer, record["prompt"], record["completion"], max_length)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    return rows, dropped


# ---- training ------------------------------------------------------------------------------------------


def _load_model(settings: SFTSettings, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    auto = AutoModelForImageTextToText if settings.image_text_model else AutoModelForCausalLM
    kwargs: dict[str, Any] = {"revision": settings.revision}
    dtype = torch.bfloat16 if settings.bf16 else torch.float32
    if settings.quantize_4bit:
        from transformers import BitsAndBytesConfig  # needs the bitsandbytes package (gpu extra)

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = dtype
    model = auto.from_pretrained(settings.model, **kwargs)
    if not settings.quantize_4bit:
        model = model.to(device)
    return model


def _pass_rate(
    model: Any, tokenizer: Any, items: Sequence[tuple[Any, Sequence[Any]]], settings: SFTSettings
) -> float:
    backend = LoadedBackend(model, tokenizer, max_new_tokens=settings.max_new_tokens)
    ok = 0
    for packet, passages in items:
        text = backend.generate(build_messages(packet, passages), n=1, temperature=0.0, seed=settings.seed)[0]
        ok += bool(check(text, packet, passages).ok)
    return ok / max(len(items), 1)


def train(
    settings: SFTSettings,
    records: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    val_items: Sequence[tuple[Any, Sequence[Any]]] = (),
) -> dict[str, Any]:
    """Fine-tune ``settings.model`` on ``records``; resume from a checkpoint in ``out_dir`` if one exists."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from fate_icu.utils.device import get_device
    from fate_icu.utils.manifest import write_run_manifest
    from fate_icu.utils.seed import seed_everything

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)
    device = get_device(settings.device)
    tokenizer = AutoTokenizer.from_pretrained(settings.model, revision=settings.revision)
    rows, dropped = tokenize_records(tokenizer, records, settings.max_length)
    if not rows:
        raise ValueError("no training example fits max_length: raise max_length or shorten the prompts")
    model = _load_model(settings, device)
    if settings.quantize_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=settings.gradient_checkpointing
        )
    selected = select_lora_modules(model)
    print(f"LoRA on {len(selected)} language-model modules (first: {selected[0]}; last: {selected[-1]})")
    lora = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=lora_target_regex(),
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps if settings.max_steps is not None else -1,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=settings.warmup_ratio,  # a float below 1 is a share of the steps
        optim=settings.optim,
        bf16=settings.bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if settings.gradient_checkpointing else None,
        logging_steps=settings.logging_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=2,
        max_length=settings.max_length,
        report_to="none",
        seed=settings.seed,
        use_cpu=device == "cpu",
        remove_unused_columns=False,
    )
    pass_rates: list[dict[str, float]] = []
    items = list(val_items)[: settings.pass_rate_packets]

    class PassRate(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **kw: Any) -> None:
            if settings.pass_rate_every and items and state.global_step % settings.pass_rate_every == 0:
                rate = _pass_rate(kw["model"], tokenizer, items, settings)
                pass_rates.append({"step": state.global_step, "pass_rate": rate})
                print(f"step {state.global_step}: validator pass rate {rate:.2f} on {len(items)} packets")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[PassRate()] if settings.pass_rate_every and items else None,
    )
    resume = any(out_dir.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.model.save_pretrained(out_dir / "adapter")
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "records": len(records),
        "trained_examples": len(rows),
        "dropped_too_long": dropped,
        "lora_modules": len(selected),
        "steps": trainer.state.global_step,
        "resumed": resume,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_loss": result.training_loss,
        "pass_rates": pass_rates,
        "device": device,
        "cuda": torch.cuda.is_available(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_manifest(
        out_dir,
        run_id=settings.run_name,
        config=settings.model_dump(),
        seeds=[settings.seed],
        model_revisions={settings.model: settings.revision or "unpinned"},
        device=device,
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True, help="SFT records (JSONL from fate_icu.llm.dataset)")
    parser.add_argument("--out", required=True, help="output folder (checkpoints, adapter, metrics.json)")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    metrics = train(settings, read_jsonl(args.records), args.out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "pass_rates"}, indent=2))


if __name__ == "__main__":
    main()
