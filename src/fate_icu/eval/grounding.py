"""Groundedness: are the claims supported by the literature they cite? (Stage 7)

An explanation is split into *claims* (its sentences, citations removed). A claim is a **patient claim**
when the
evidence packet entails it, and an **external claim** otherwise: general medical knowledge, or an
invention. The
packet is checked first so that the same rule works for a model that never saw passages (E6) and one that
did (E7),
without guessing from the wording which sentences "sound medical".

Metrics, following ALCE (Gao et al., 2023) with the claims restricted to the external ones:

* ``citation_recall``: share of external claims entailed by the concatenation of the passages they cite
  (a claim that cites no passage counts as unsupported);
* ``citation_precision``: over all passage citations, the share that are *needed*: the claims's cited set
  entails it
  and the passage alone entails it, or removing the passage makes the rest fail to entail it;
* ``pool_support``: share of external claims entailed by at least one passage of the stay's whole
  retrieved pool,
  cited or not; ``1 - pool_support`` is the unsupported-claim rate. This is the H3 metric, since E6 cites
  nothing.

The entailment judge is a ``Checker`` (``entails(premise, claim) -> bool``): ``NLIChecker`` wraps a
Hugging Face NLI
model (a claim is supported when the most probable label is entailment: the threshold registered in
``docs/hypotheses.md``); ``LexicalChecker`` is a deterministic word-overlap stand-in for tests and smoke
runs only.
``cohens_kappa`` and ``checker_agreement`` measure the checker against the user's hand-labelled pairs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fate_icu.llm.prompt import render_evidence
from fate_icu.llm.validate import CITATION, split_sentences

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class Checker(Protocol):
    def entails(self, premise: str, claim: str) -> bool: ...


@dataclass(frozen=True)
class Claim:
    text: str  # the sentence without its citation tags
    driver_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]


def split_claims(text: str) -> list[Claim]:
    """The claims of an explanation: one per sentence, with the ids it cites."""
    claims = []
    for sentence in split_sentences(text):
        ids = CITATION.findall(sentence)
        clean = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITATION.sub("", sentence).split()))
        claims.append(
            Claim(
                clean,
                tuple(i for i in ids if i.startswith("D")),
                tuple(dict.fromkeys(i for i in ids if i.startswith("R"))),
            )
        )
    return claims


def _passage_text(passages: Sequence[Any], rid: str) -> str | None:
    for p in passages:
        if (p.get("rid") if isinstance(p, dict) else getattr(p, "rid", None)) == rid:
            return str(p.get("text") if isinstance(p, dict) else p.text)
    return None


def claim_counts(text: str, packet: Any, passages: Sequence[Any], checker: Checker) -> dict[str, int]:
    """The counts behind every groundedness metric of one explanation.

    Keys: ``n_claims``, ``n_external`` (not entailed by the packet), ``n_cited_supported`` (external
    claims entailed by
    their cited passages), ``n_pool_supported`` (external claims entailed by some passage of the pool),
    ``n_citations``
    (passage citations on external claims) and ``n_citations_needed``.
    """
    premise = render_evidence(packet)
    out = dict.fromkeys(
        (
            "n_claims",
            "n_external",
            "n_cited_supported",
            "n_pool_supported",
            "n_citations",
            "n_citations_needed",
        ),
        0,
    )
    for claim in split_claims(text):
        out["n_claims"] += 1
        if not claim.text or checker.entails(premise, claim.text):
            continue  # a statement of the patient's own evidence, supported by the packet
        out["n_external"] += 1
        cited = [(rid, _passage_text(passages, rid)) for rid in claim.passage_ids]
        cited = [(rid, t) for rid, t in cited if t is not None]
        if any(checker.entails(str(_get(p, "text")), claim.text) for p in passages):
            out["n_pool_supported"] += 1
        if not cited:
            continue
        joined = " ".join(t for _, t in cited)
        supported = checker.entails(joined, claim.text)
        out["n_cited_supported"] += supported
        out["n_citations"] += len(cited)
        if not supported:
            continue  # the cited set does not entail the claim: none of its citations is correct
        for rid, passage in cited:
            if checker.entails(passage, claim.text):
                out["n_citations_needed"] += 1
                continue
            rest = " ".join(t for r, t in cited if r != rid)
            if not rest or not checker.entails(rest, claim.text):  # the claim needs this passage in the set
                out["n_citations_needed"] += 1
    return out


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name)


def rates(counts: dict[str, int]) -> dict[str, float | None]:
    """The metrics from the counts of one explanation, or from the sum of many (``None``: undefined)."""

    def share(a: str, b: str) -> float | None:
        return counts[a] / counts[b] if counts[b] else None

    pool = share("n_pool_supported", "n_external")
    return {
        "citation_recall": share("n_cited_supported", "n_external"),
        "citation_precision": share("n_citations_needed", "n_citations"),
        "pool_support": pool,
        "unsupported_claim_rate": None if pool is None else 1.0 - pool,
    }


# ---- agreement with hand labels --------------------------------------------------------------------


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa of two labellings of the same items (``nan`` when chance agreement is 1)."""
    if len(a) != len(b) or not len(a):
        raise ValueError("the two labellings must be non-empty and of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((sum(x == k for x in a) / n) * (sum(y == k for y in b) / n) for k in set(a) | set(b))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def checker_agreement(pairs: Sequence[dict[str, Any]], checker: Checker) -> dict[str, float | int]:
    """Agreement of the checker with hand labels: each pair has ``premise``, ``claim`` and ``supported``
    (bool)."""
    human = [bool(p["supported"]) for p in pairs]
    machine = [bool(checker.entails(p["premise"], p["claim"])) for p in pairs]
    return {
        "n": len(pairs),
        "kappa": cohens_kappa(human, machine),
        "accuracy": sum(h == m for h, m in zip(human, machine, strict=True)) / len(pairs),
    }


# ---- checkers --------------------------------------------------------------------------------------

_STOP = frozenset(
    [
        "with",
        "that",
        "this",
        "from",
        "have",
        "were",
        "been",
        "will",
        "also",
        "than",
        "they",
        "them",
        "then",
        "which",
        "when",
        "what",
        "there",
        "their",
        "about",
        "into",
        "over",
        "more",
        "most",
        "some",
        "such",
        "only",
        "both",
        "each",
        "other",
        "these",
        "those",
        "while",
        "after",
        "before",
        "between",
        "during",
        "under",
        "again",
    ]
)
# Words that frame a prediction or a driver sentence rather than assert a fact ("the model predicts ... mainly
# because of ... points toward ... smaller contributions came from ..."): ignored by the stand-in checker.
_FRAMING = frozenset(
    [
        "model",
        "predicts",
        "predicted",
        "probability",
        "mainly",
        "because",
        "points",
        "toward",
        "smaller",
        "contributions",
        "came",
        "leans",
    ]
)


class LexicalChecker:
    """A claim is entailed when every content word (4+ letters, not a stop word) of it is in the premise.

    Deterministic and offline: for tests and smoke runs. It is not an NLI model and must not back any
    reported number.
    """

    def entails(self, premise: str, claim: str) -> bool:
        words = {
            w for w in re.findall(r"[a-z]{4,}", claim.casefold()) if w not in _STOP and w not in _FRAMING
        }
        have = set(re.findall(r"[a-z]{4,}", premise.casefold()))
        return bool(words) and words <= have


class NLIChecker:
    """A Hugging Face NLI model: a claim is supported when the most probable label is entailment.

    The model and tokenizer load on first use (``transformers`` imports are lazy). ``model`` may be a folder.
    """

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name, self.revision, self.batch_size, self.max_length = (
            model,
            revision,
            batch_size,
            max_length,
        )
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._entailment: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name, revision=self.revision)
            .to(self.device)
            .eval()
        )
        labels = {int(i): str(name).casefold() for i, name in self._model.config.id2label.items()}
        matches = [i for i, name in labels.items() if name.startswith("entail")]
        if len(matches) != 1:
            raise ValueError(f"cannot find the entailment label in {labels}")
        self._entailment = matches[0]

    def entails_many(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        import torch

        self._load()
        out: list[bool] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            batch = self._tokenizer(
                [p for p, _ in chunk],
                [c for _, c in chunk],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**batch).logits
            out += (logits.argmax(dim=-1) == self._entailment).tolist()
        return out

    def entails(self, premise: str, claim: str) -> bool:
        return self.entails_many([(premise, claim)])[0]
