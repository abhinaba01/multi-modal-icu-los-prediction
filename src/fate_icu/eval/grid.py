"""The ablation grid of Stage 7: which predictor, which packets and which explainer each row uses.

Every row is evaluated on identical test patients. A row is *runnable* when the artifacts it needs exist:
a predictor arm (Stage 3), attribution packets (Stage 4), passages (Stage 5) and, for the language-model
rows, generated texts (Stage 6). ``missing`` lists what a run lacks, so a partial grid reports the rows it
can and says which it could not, instead of silently dropping them.

Text systems: ``template`` is the deterministic explanation of ``llm/template.py``; ``zero_shot`` is MedGemma
with the strict prompt and two worked examples; ``sft`` is the QLoRA-tuned model; ``sft_dpo`` adds DPO.
Packet sources: ``attribution`` (Stage 4), ``summary`` (control C1: a deterministic summary of every
abnormal value) and ``random`` (control C2: random units, truthful descriptions, no attribution).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GridRow:
    id: str
    name: str
    predictor: str  # a Stage 3 arm
    packets: str | None  # "attribution" | "summary" | "random" | "shap" | None (predictor only)
    text: str | None  # None | "template" | "zero_shot" | "sft" | "sft_dpo"
    rag: bool = False
    control: bool = False


GRID: tuple[GridRow, ...] = (
    GridRow("E0", "internship XGBoost + SHAP feature list", "B0", "shap", None),
    GridRow("E1", "EHR-only neural model (static + time series)", "B3", "attribution", None),
    GridRow("E2", "image only", "B4", "attribution", None),
    GridRow("E3", "fusion (ours)", "M", "attribution", None),
    GridRow("E4", "E3 + template explanation", "M", "attribution", "template"),
    GridRow("E5", "E3 + zero-shot MedGemma, packet-constrained, no RAG", "M", "attribution", "zero_shot"),
    GridRow("E6", "E3 + SFT MedGemma, no RAG", "M", "attribution", "sft"),
    GridRow("E7", "E3 + SFT MedGemma + RAG", "M", "attribution", "sft", rag=True),
    GridRow("E8", "E7 + DPO", "M", "attribution", "sft_dpo", rag=True),
    GridRow("C1", "control: zero-shot MedGemma, abnormal values", "M", "summary", "zero_shot", control=True),
    GridRow("C2", "control: SFT + RAG, random drivers", "M", "random", "sft", rag=True, control=True),
)  # fmt: skip
BY_ID = {row.id: row for row in GRID}


def missing(
    row: GridRow,
    arms: Collection[str],
    packets: Collection[str],
    texts: Collection[str],
    has_passages: bool,
) -> list[str]:
    """What this row needs that the run does not have (empty: runnable)."""
    out = []
    if row.predictor not in arms:
        out.append(f"predictor arm {row.predictor}")
    if row.packets and row.packets not in packets:
        out.append(f"{row.packets} packets")
    if row.text and row.id not in texts:
        out.append(f"generated texts for {row.id}")
    if row.rag and not has_passages:
        out.append("retrieved passages")
    return out


def runnable_rows(
    arms: Collection[str],
    packets: Collection[str],
    texts: Collection[str],
    has_passages: bool,
    rows: Sequence[GridRow] = GRID,
) -> tuple[list[GridRow], dict[str, list[str]]]:
    """The rows that can be evaluated, and for each other row what it lacks."""
    ok, lacking = [], {}
    for row in rows:
        gaps = missing(row, arms, packets, texts, has_passages)
        if gaps:
            lacking[row.id] = gaps
        else:
            ok.append(row)
    return ok, lacking


def grid_table(rows: Sequence[GridRow] = GRID) -> list[dict[str, str]]:
    """The grid as table rows (strings), for the report."""
    return [
        {
            "id": r.id,
            "configuration": r.name,
            "predictor": r.predictor,
            "packets": r.packets or "-",
            "text": r.text or "-",
            "rag": "yes" if r.rag else "no",
            "control": "yes" if r.control else "no",
        }
        for r in rows
    ]
