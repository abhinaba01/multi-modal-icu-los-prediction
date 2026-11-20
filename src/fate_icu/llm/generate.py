"""Generating an explanation: ask the model, check the contract, resample, fall back (Stage 6).

``explain_one`` asks a backend for an explanation of one packet. The first try is greedy and deterministic;
if it breaks the contract of ``validate.check`` the backend is asked again at a higher temperature, up to
``max_resamples`` more times, and if every try fails the deterministic template of ``template.py`` is used.
The text that is returned therefore always meets the contract, and ``RunStats`` counts how often a retry or
the fallback was needed, which is reported and never hidden.

A *backend* is anything with ``generate(messages, n, temperature, seed) -> list[str]``. ``HFBackend`` is the
Hugging Face one (imports are lazy, so the module loads without a GPU or the model); tests use a scripted
one.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from fate_icu.llm.prompt import build_messages
from fate_icu.llm.template import explain as template_explain
from fate_icu.llm.validate import check

MAX_RESAMPLES = 3
RESAMPLE_TEMPERATURE = 0.7
_PREFIX = re.compile(r"^\s*(?:explanation\s*:\s*)", re.IGNORECASE)


class Backend(Protocol):
    def generate(
        self, messages: list[dict[str, str]], *, n: int = 1, temperature: float = 0.0, seed: int | None = None
    ) -> list[str]: ...


@dataclass(frozen=True)
class Generated:
    text: str
    source: str  # "model" or "template"
    attempts: int  # model calls made (a fallback after three failed resamples has 4)
    failures: tuple[tuple[str, ...], ...]  # the broken rules of each failed attempt, in order


@dataclass
class RunStats:
    n: int = 0
    retried: int = 0  # needed more than one model call
    fell_back: int = 0  # used the template
    rules: Counter = field(default_factory=Counter)  # how often each rule was broken (over all attempts)

    def add(self, result: Generated) -> None:
        self.n += 1
        self.retried += result.attempts > 1
        self.fell_back += result.source == "template"
        for rules in result.failures:
            self.rules.update(rules)

    def rates(self) -> dict[str, Any]:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "retry_rate": self.retried / n,
            "fallback_rate": self.fell_back / n,
            "rule_failures": dict(self.rules),
        }


def clean(text: str) -> str:
    """The explanation without a leading "Explanation:" label or surrounding whitespace."""
    return _PREFIX.sub("", text.strip()).strip()


def explain_one(
    backend: Backend,
    packet: Any,
    passages: Sequence[Any] = (),
    *,
    max_resamples: int = MAX_RESAMPLES,
    resample_temperature: float = RESAMPLE_TEMPERATURE,
    seed: int = 0,
    few_shot: bool = False,
) -> Generated:
    """An explanation of ``packet`` that meets the contract (the template if the model never does)."""
    messages = build_messages(packet, passages, few_shot=few_shot)
    failures: list[tuple[str, ...]] = []
    for attempt in range(max_resamples + 1):
        temperature = 0.0 if attempt == 0 else resample_temperature
        text = clean(backend.generate(messages, n=1, temperature=temperature, seed=seed + attempt)[0])
        verdict = check(text, packet, passages)
        if verdict.ok:
            return Generated(text, "model", attempt + 1, tuple(failures))
        failures.append(verdict.failures)
    return Generated(template_explain(packet), "template", max_resamples + 1, tuple(failures))


def explain_many(
    backend: Backend, items: Sequence[tuple[Any, Sequence[Any]]], *, seed: int = 0, **kwargs: Any
) -> tuple[list[Generated], RunStats]:
    """``explain_one`` for each (packet, passages), in order, with the retry and fallback counts.

    Each packet gets its own block of seeds, so a result does not depend on the other packets.
    """
    stats = RunStats()
    results = []
    block = kwargs.get("max_resamples", MAX_RESAMPLES) + 1
    for index, (packet, passages) in enumerate(items):
        result = explain_one(backend, packet, passages, seed=seed + index * block, **kwargs)
        stats.add(result)
        results.append(result)
    return results, stats


class HFBackend:
    """A Hugging Face chat model, optionally with a LoRA adapter. Needs ``transformers`` (and ``peft`` for an
    adapter); weights are loaded on first use. ``temperature == 0`` means greedy decoding."""

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        adapter: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 320,
        image_text_model: bool = False,
    ):
        self.model_name, self.revision, self.adapter = model, revision, adapter
        self.device, self.max_new_tokens, self.image_text_model = device, max_new_tokens, image_text_model
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        auto = AutoModelForImageTextToText if self.image_text_model else AutoModelForCausalLM
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        model = auto.from_pretrained(self.model_name, revision=self.revision, dtype=dtype)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        self._model = model.to(device).eval()
        self.device = device

    def generate(
        self, messages: list[dict[str, str]], *, n: int = 1, temperature: float = 0.0, seed: int | None = None
    ) -> list[str]:
        self._load()
        return chat_generate(
            self._model,
            self._tokenizer,
            messages,
            n=n,
            temperature=temperature,
            seed=seed,
            max_new_tokens=self.max_new_tokens,
        )


class LoadedBackend:
    """A backend around a model and tokenizer that are already in memory (e.g. the model being trained)."""

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 320):
        self.model, self.tokenizer, self.max_new_tokens = model, tokenizer, max_new_tokens

    def generate(
        self, messages: list[dict[str, str]], *, n: int = 1, temperature: float = 0.0, seed: int | None = None
    ) -> list[str]:
        return chat_generate(
            self.model,
            self.tokenizer,
            messages,
            n=n,
            temperature=temperature,
            seed=seed,
            max_new_tokens=self.max_new_tokens,
        )


def chat_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    n: int = 1,
    temperature: float = 0.0,
    seed: int | None = None,
    max_new_tokens: int = 320,
) -> list[str]:
    """``n`` replies of ``model`` to the chat ``messages``; ``temperature == 0`` means greedy decoding."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    sample = temperature > 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=sample,
                temperature=temperature if sample else None,
                top_p=0.95 if sample else None,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()
    new = out[:, inputs["input_ids"].shape[1] :]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in new]
