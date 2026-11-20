"""Training data for the explainer: rejection-sampling SFT records and DPO preference pairs (Stage 6).

For each (packet, passages) the base model is sampled several times with a strict few-shot prompt. A sample
is *kept* when it meets the contract of ``validate.check`` and, when given, a second checker
(``extra_check``: the training-time support checker, which must differ from the evaluation checker of
Stage 7). Kept samples become SFT records. DPO pairs prefer a kept sample over (a) a sample that failed, and
(b) a *corrupted* copy of the kept sample: a swapped driver, a flipped direction or a mis-cited passage.
Every corrupted text is verified to break the contract, so a pair never prefers a text that is also valid.

Records use TRL's conversational prompt-completion format. The prompt of a record has no worked examples (the
tuned model learns the format; ``prompt.build_messages(few_shot=False)``), although sampling used them.
Everything here is plain Python on strings; the data it makes is credentialed when the packets are, so it is
written only under the artifacts directory of the GCP VM.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fate_icu.llm.generate import Backend, clean
from fate_icu.llm.prompt import build_messages
from fate_icu.llm.validate import CITATION, check, split_sentences

SAMPLES_PER_PROMPT = 8
SAMPLING_TEMPERATURE = 0.8
MAX_KEPT_PER_PROMPT = 2
CORRUPTIONS = ("flip_direction", "swap_driver", "miscite")
ExtraCheck = Callable[[str, Any, Sequence[Any]], bool]


@dataclass(frozen=True)
class Candidate:
    text: str
    ok: bool
    failures: tuple[str, ...]


def _driver_ids(packet: Any) -> list[str]:
    drivers = packet["drivers"] if isinstance(packet, dict) else packet.drivers
    return [d["id"] if isinstance(d, dict) else d.id for d in drivers]


def sample_candidates(
    backend: Backend,
    packet: Any,
    passages: Sequence[Any] = (),
    *,
    n: int = SAMPLES_PER_PROMPT,
    temperature: float = SAMPLING_TEMPERATURE,
    seed: int = 0,
    few_shot: bool = True,
    extra_check: ExtraCheck | None = None,
) -> list[Candidate]:
    """``n`` samples for one prompt, each checked against the contract (and ``extra_check`` when given)."""
    messages = build_messages(packet, passages, few_shot=few_shot)
    out = []
    for text in backend.generate(messages, n=n, temperature=temperature, seed=seed):
        text = clean(text)
        verdict = check(text, packet, passages)
        failures = verdict.failures
        if verdict.ok and extra_check is not None and not extra_check(text, packet, passages):
            failures = ("support_check",)
        out.append(Candidate(text, not failures, failures))
    return out


# ---- corrupted negatives ------------------------------------------------------------------------------


def _replace_in_sentence(text: str, index: int, new: str) -> str:
    sentences = split_sentences(text)
    sentences[index] = new
    return " ".join(sentences)


def corrupt(text: str, packet: Any, kind: str, rng: random.Random) -> str | None:
    """A copy of ``text`` with one fault of ``kind``, or ``None`` when the text has nothing to corrupt."""
    sentences = split_sentences(text)
    ids = _driver_ids(packet)
    if kind == "flip_direction":
        spots = [i for i, s in enumerate(sentences) if i > 0 and re.search(r"\b(longer|shorter)\b", s)]
        if not spots:
            return None
        i = rng.choice(spots)
        flipped = re.sub(
            r"\b(longer|shorter)\b",
            lambda m: "shorter" if m.group(1) == "longer" else "longer",
            sentences[i],
        )
        return _replace_in_sentence(text, i, flipped)
    if kind == "swap_driver":
        spots = [(i, c) for i, s in enumerate(sentences) for c in CITATION.findall(s) if c in ids]
        if not spots or len(ids) < 2:
            return None
        i, cited = rng.choice(spots)
        other = rng.choice([d for d in ids if d != cited])
        return _replace_in_sentence(text, i, sentences[i].replace(f"[{cited}]", f"[{other}]"))
    if kind == "miscite":
        spots = [(i, c) for i, s in enumerate(sentences) for c in CITATION.findall(s)]
        if not spots:
            return None
        i, cited = rng.choice(spots)
        unknown = "R9" if cited.startswith("R") else "D9"
        return _replace_in_sentence(text, i, sentences[i].replace(f"[{cited}]", f"[{unknown}]"))
    raise ValueError(f"unknown corruption {kind!r}; choose from {CORRUPTIONS}")


def corrupted_negatives(
    text: str, packet: Any, passages: Sequence[Any], rng: random.Random
) -> list[tuple[str, str]]:
    """``(kind, corrupted text)`` for each corruption that applies and verifiably breaks the contract."""
    out = []
    for kind in CORRUPTIONS:
        bad = corrupt(text, packet, kind, rng)
        if bad is not None and bad != text and not check(bad, packet, passages).ok:
            out.append((kind, bad))
    return out


# ---- records ----------------------------------------------------------------------------------------


def _assistant(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}]


@dataclass
class DatasetStats:
    prompts: int = 0
    samples: int = 0
    samples_ok: int = 0
    prompts_without_a_kept_sample: int = 0
    sft_records: int = 0
    dpo_pairs: Counter | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompts": self.prompts,
            "samples": self.samples,
            "sample_pass_rate": self.samples_ok / max(self.samples, 1),
            "prompts_without_a_kept_sample": self.prompts_without_a_kept_sample,
            "sft_records": self.sft_records,
            "dpo_pairs": dict(self.dpo_pairs or {}),
        }


def build_records(
    items: Iterable[tuple[Any, Sequence[Any]]],
    backend: Backend,
    *,
    n: int = SAMPLES_PER_PROMPT,
    temperature: float = SAMPLING_TEMPERATURE,
    max_kept: int = MAX_KEPT_PER_PROMPT,
    seed: int = 0,
    extra_check: ExtraCheck | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], DatasetStats]:
    """SFT records and DPO pairs from sampling ``backend`` on every (packet, passages)."""
    rng = random.Random(seed)
    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    stats = DatasetStats(dpo_pairs=Counter())
    for index, (packet, passages) in enumerate(items):
        candidates = sample_candidates(
            backend,
            packet,
            passages,
            n=n,
            temperature=temperature,
            seed=seed + index * n,
            extra_check=extra_check,
        )
        stats.prompts += 1
        stats.samples += len(candidates)
        kept = list(dict.fromkeys(c.text for c in candidates if c.ok))
        stats.samples_ok += sum(c.ok for c in candidates)
        if not kept:
            stats.prompts_without_a_kept_sample += 1
            continue
        prompt = build_messages(packet, passages, few_shot=False)
        failed = [c for c in candidates if not c.ok]
        for text in kept[:max_kept]:
            sft.append({"prompt": prompt, "completion": _assistant(text)})
            if failed:
                loser = rng.choice(failed)
                dpo.append(_pair(prompt, text, loser.text, "failed_sample"))
                stats.dpo_pairs["failed_sample"] += 1
            for kind, bad in corrupted_negatives(text, packet, passages, rng):
                dpo.append(_pair(prompt, text, bad, kind))
                stats.dpo_pairs[kind] += 1
    stats.sft_records = len(sft)
    return sft, dpo, stats


def _pair(prompt: list[dict[str, str]], chosen: str, rejected: str, kind: str) -> dict[str, Any]:
    return {"prompt": prompt, "chosen": _assistant(chosen), "rejected": _assistant(rejected), "kind": kind}


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
