"""A synthetic evaluation bundle for the Stage 7 smoke run (no patient data, no model).

It builds, for ``n`` fake test stays: the predictions of four arms (M a little better than B3 when paired),
explained stays for the grid rows E4 to E8, C1 and C2 with texts that differ in the ways the hypotheses look
for, and per-stay selection material for H4 (a toy linear model). The texts are made by rules, so the numbers
the evaluation computes from them show only that the code works end to end, never how a real system behaves.
"""

from __future__ import annotations

import random

import numpy as np

from fate_icu.eval.controls import random_packet
from fate_icu.eval.run import ArmPredictions, EvalBundle, ExplainedItem
from fate_icu.eval.selection import StaySelection
from fate_icu.llm.synthetic import random_items
from fate_icu.llm.template import explain

FABRICATED = "Vitamin supplements reverse organ failure completely."
SAMPLE = 200  # explained stays (the registered sample is 1,000 on real data)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _arms(n: int, rng: np.random.Generator) -> dict[str, ArmPredictions]:
    y = (rng.random(n) < 0.3).astype(int)
    patients = np.asarray([f"p{i // 2}" for i in range(n)])
    paired = rng.random(n) < 0.25
    los = np.where(y == 1, rng.uniform(4.2, 12.0, n), rng.uniform(0.5, 4.0, n))
    out = {}
    for arm, sep in {"B0": 0.6, "B3": 1.0, "B4": 0.5, "M": 1.0}.items():
        sep_row = np.where(paired, sep + (0.35 if arm == "M" else 0.0), sep)
        p = _sigmoid(sep_row * 2.0 * (y - 0.3) + rng.normal(0, 1, n))
        out[arm] = ArmPredictions(
            y, p, los, np.clip(los + rng.normal(0, 1.0, n), 0.1, None), patients, paired
        )
    return out


def _cited(sentence: str, rid: str = "R1") -> str:
    """The sentence with its citation before the final period, as the output contract has it."""
    return f"{sentence.rstrip('.')} [{rid}]."


def _passage_sentence(passage: dict) -> str:
    text = passage["text"].replace("Synthetic passage: ", "")
    return text[:1].upper() + text[1:]


def _selection(rng: np.random.Generator, n: int) -> list[StaySelection]:
    stays = []
    for _ in range(n):
        w = rng.uniform(0.0, 2.0, 8)
        stays.append(
            StaySelection(
                n_units=8,
                score_removed=lambda masks, w=w: 0.5 + ((~np.asarray(masks, dtype=bool)) * w).sum(axis=1),
                two_step=np.argsort(-w)[:3],
                gradient_scores=w + rng.normal(0, 1.0, 8),
            )
        )
    return stays


def synthetic_bundle(n: int = 300, seed: int = 666) -> EvalBundle:
    rng, pyrng = np.random.default_rng(seed), random.Random(seed)
    arms = _arms(n, rng)
    m = min(n, SAMPLE)
    items = random_items(m, seed=seed)
    pool = [
        {k: v for k, v in d.items() if k not in ("id", "effect")}
        for packet, _ in items
        for d in packet["drivers"]
    ]
    rows: dict[str, list[ExplainedItem]] = {r: [] for r in ("E4", "E5", "E6", "E7", "E8", "C1", "C2")}
    for i, (packet, passages) in enumerate(items):
        stay, patient = f"s{i}", f"p{i // 2}"
        effect = {d["id"]: d["effect"] for d in packet["drivers"]}

        def item(text, packet=packet, passages=passages, stay=stay, patient=patient, effect=effect):
            return ExplainedItem(
                stay, patient, packet, passages, text, delta_of=lambda driver_id, e=effect: e[driver_id]
            )

        base = explain(packet)
        general = _passage_sentence(passages[0]) if passages else None
        rows["E4"].append(item(base))
        extra = " Serum lactate also played a role [D1]." if pyrng.random() < 0.15 else ""
        rows["E5"].append(item(base + extra))
        c1_text = explain(random_packet(packet, pool, pyrng)) if len(pool) >= len(packet["drivers"]) else base
        rows["C1"].append(item(c1_text))
        rows["C2"].append(item(c1_text))
        if general:
            e6 = general if pyrng.random() < 0.5 else FABRICATED
            e7 = _cited(general if pyrng.random() < 0.9 else FABRICATED)
            e8 = _cited(general if pyrng.random() < 0.95 else FABRICATED)
            rows["E6"].append(item(f"{base} {e6}"))
            rows["E7"].append(item(f"{base} {e7}"))
            rows["E8"].append(item(f"{base} {e8}"))
        else:
            for r in ("E6", "E7", "E8"):
                rows[r].append(item(base))
    k = min(n, 150)
    return EvalBundle(
        arms=arms,
        explained=rows,
        selection=_selection(rng, k),
        selection_patients=np.asarray([f"p{i // 2}" for i in range(k)]),
    )
