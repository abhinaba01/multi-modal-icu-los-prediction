"""The Stage 7 evaluation: one command, one report.

    python -m fate_icu.eval.run --config configs/eval/smoke.yaml

An *evaluation bundle* holds, for the same test patients: the predictions of the Stage 3 arms, the explanation
of each grid row for a sample of stays (with the stay's attribution packet and retrieved passages), and the
per-stay material of H4. ``evaluate`` turns a bundle into metrics with patient-level bootstrap intervals and
tests the registered hypotheses (``docs/hypotheses.md``) with Holm-Bonferroni across the ones that can be
tested; ``write_report`` writes ``<reports>/<run_id>/`` with ``metrics.json``, CSV and Markdown tables and
``run_manifest.json``. A hypothesis or grid row whose inputs are missing is reported as *not run*, with the
reason; it is never silently dropped or filled in.

The smoke config evaluates a synthetic bundle (``smoke_data.py``) with the offline word-overlap checker: it
proves that the code runs end to end and says nothing about any real system.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from fate_icu.eval import explanation as X
from fate_icu.eval import grounding as G
from fate_icu.eval.grid import runnable_rows
from fate_icu.eval.predictive import auprc, auroc, evaluate_population
from fate_icu.eval.selection import StaySelection, h4_test, per_stay_values
from fate_icu.eval.stats import (
    holm_bonferroni,
    mean_ci,
    paired_mean_difference,
    paired_patient_bootstrap,
    ratio_ci,
)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    source: Literal["synthetic", "vm"] = "synthetic"
    n_stays: int = 300  # synthetic bundle size
    n_boot: int = 1000
    seed: int = 666
    k: int = 3  # driver recall@k and H4's k
    tau: float = 0.1  # claim deletion: the smallest effect that counts as moving the logit
    checker: Literal["lexical", "nli"] = "lexical"
    nli_model: str = G.DEFAULT_NLI_MODEL
    nli_revision: str | None = None
    alpha: float = 0.05
    figures: bool = False
    reports_dir: str | None = None  # default: <outputs>/../reports from the settings


def load_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class ArmPredictions:
    y_long: np.ndarray
    p_long: np.ndarray  # seed-ensembled probability of a long stay
    los_days: np.ndarray
    pred_los_days: np.ndarray
    patients: np.ndarray
    paired: np.ndarray  # bool: the stay has an eligible X-ray before t0 (population P1)


@dataclass
class ExplainedItem:
    stay: str  # the same key in every row of the grid, so rows can be paired
    patient: str
    packet: Any  # the stay's attribution packet: the reference every text is scored against
    passages: Sequence[Any]
    text: str
    delta_of: Callable[[str], float] | None = None  # claim deletion; needs the model


@dataclass
class EvalBundle:
    arms: dict[str, ArmPredictions]
    explained: dict[str, list[ExplainedItem]]  # grid row id -> its explained stays
    selection: list[StaySelection] = field(default_factory=list)
    selection_patients: np.ndarray = field(default_factory=lambda: np.array([]))


def _key(item: ExplainedItem) -> str:
    return item.stay


def _per_item_metrics(
    items: Sequence[ExplainedItem], checker: G.Checker, lexicon: X.Lexicon, config: EvalConfig
) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        m = X.per_stay_metrics(it.text, it.packet, it.passages, lexicon, k=config.k)
        if it.delta_of is not None:
            m["claim_deletion_validity"] = X.claim_deletion_validity(
                it.text, it.packet, lexicon, it.delta_of, config.tau
            )
        counts = G.claim_counts(it.text, it.packet, it.passages, checker)
        rows.append({**m, **{f"count_{k}": v for k, v in counts.items()}, **G.rates(counts)})
    return rows


def _column(rows: Sequence[Mapping[str, Any]], name: str) -> list[float | None]:
    return [r.get(name) for r in rows]


EXPLANATION_METRICS = (
    "driver_precision",
    "unattributed_rate",
    "driver_recall_at_3",
    "direction_accuracy",
    "numeric_consistency",
    "claim_deletion_validity",
)
COUNT_KEYS = ("n_external", "n_pool_supported", "n_cited_supported", "n_citations", "n_citations_needed")


def _paired(a_rows, b_rows, a_items, b_items, name):
    """Per-stay values of ``name`` for the stays in both rows, in the same order, with their patients."""
    index = {_key(i): n for n, i in enumerate(b_items)}
    va, vb, pat = [], [], []
    for n, item in enumerate(a_items):
        m = index.get(_key(item))
        if m is None:
            continue
        x, y = a_rows[n].get(name), b_rows[m].get(name)
        if x is None or y is None:
            continue
        va.append(x)
        vb.append(y)
        pat.append(item.patient)
    return np.asarray(va, float), np.asarray(vb, float), np.asarray(pat)


def evaluate(
    bundle: EvalBundle, config: EvalConfig, checker: G.Checker, lexicon: X.Lexicon
) -> dict[str, Any]:
    """Every metric, every registered test, and what could not be run."""
    out: dict[str, Any] = {
        "config": config.model_dump(),
        "predictive": {},
        "explanation": {},
        "grounding": {},
    }
    boot = {"n_boot": config.n_boot, "seed": config.seed}

    for arm, p in bundle.arms.items():
        out["predictive"][arm] = {
            "all_test_stays": evaluate_population(
                p.y_long, p.p_long, p.los_days, p.pred_los_days, p.patients, **boot
            ),
            "n_stays": int(len(p.y_long)),
            "n_paired_stays": int(p.paired.sum()),
        }

    rows_metrics: dict[str, list[dict[str, Any]]] = {}
    for row_id, items in bundle.explained.items():
        rows = _per_item_metrics(items, checker, lexicon, config)
        rows_metrics[row_id] = rows
        patients = np.asarray([i.patient for i in items])
        out["explanation"][row_id] = {
            m: mean_ci(_column(rows, m), patients, **boot)
            for m in EXPLANATION_METRICS
            if any(m in r for r in rows)
        }
        totals = {k: int(sum(r[f"count_{k}"] for r in rows)) for k in COUNT_KEYS}
        ratios = {}
        for name, (num, den) in {
            "citation_recall": ("n_cited_supported", "n_external"),
            "citation_precision": ("n_citations_needed", "n_citations"),
            "pool_support": ("n_pool_supported", "n_external"),
        }.items():
            ratios[name] = ratio_ci(
                np.asarray([r[f"count_{num}"] for r in rows]),
                np.asarray([r[f"count_{den}"] for r in rows]),
                patients,
                **boot,
            )
        ratios["unsupported_claim_rate"] = {
            k: (None if v is None else 1.0 - v) if k == "ratio" else v
            for k, v in ratios["pool_support"].items()
        }
        ratios["unsupported_claim_rate"]["low"], ratios["unsupported_claim_rate"]["high"] = (
            1.0 - ratios["pool_support"]["high"],
            1.0 - ratios["pool_support"]["low"],
        )
        out["grounding"][row_id] = {"counts": totals, **ratios}

    tests: dict[str, dict[str, Any]] = {}
    not_run: dict[str, str] = {}

    # H1: M vs B3 on AUPRC, population P1
    if {"M", "B3"} <= set(bundle.arms):
        m, b3 = bundle.arms["M"], bundle.arms["B3"]
        keep = m.paired
        tests["H1"] = {
            **paired_patient_bootstrap(
                auprc, m.y_long[keep], m.p_long[keep], b3.p_long[keep], m.patients[keep], **boot
            ),
            "n_stays": int(keep.sum()),
            "metric": "AUPRC(M) - AUPRC(B3) on P1",
            "expected_sign": "positive",
        }
        tests["H1"]["secondary_auroc"] = paired_patient_bootstrap(
            auroc, m.y_long[keep], m.p_long[keep], b3.p_long[keep], m.patients[keep], **boot
        )
    else:
        not_run["H1"] = "needs the predictions of arms M and B3"

    items = bundle.explained

    def _diff(first, second, name):
        if first not in items or second not in items:
            return None
        a, b, pat = _paired(rows_metrics[first], rows_metrics[second], items[first], items[second], name)
        return paired_mean_difference(a, b, pat, **boot) | {"n_stays": int(len(a))} if len(a) else None

    h2 = _diff("E5", "C1", "unattributed_rate")
    if h2:
        tests["H2"] = h2 | {
            "metric": "unattributed_rate(E5) - unattributed_rate(C1)",
            "expected_sign": "negative",
        }
    else:
        not_run["H2"] = "needs the explained stays of E5 and C1"
    h3 = _diff("E7", "E6", "pool_support")
    if h3:
        tests["H3"] = h3 | {"metric": "pool_support(E7) - pool_support(E6)", "expected_sign": "positive"}
    else:
        not_run["H3"] = "needs the explained stays of E7 and E6"

    if bundle.selection:
        values = per_stay_values(bundle.selection, k=config.k, seed=config.seed)
        h4 = h4_test(values["comprehensiveness"], bundle.selection_patients, **boot)
        h4["metric"] = f"comprehensiveness@{config.k}: two-step minus gradient-only and minus random"
        h4["expected_sign"] = "positive"
        h4["means"] = {k: float(v.mean()) for k, v in values["comprehensiveness"].items()}
        tests["H4"] = h4
    else:
        not_run["H4"] = "needs per-stay selection material (the explained model's removal scores)"

    if "E8" in items and "E7" in items:
        parts = {}
        for name in ("driver_precision", "citation_precision", "driver_recall_at_3"):
            a, b, pat = _paired(rows_metrics["E8"], rows_metrics["E7"], items["E8"], items["E7"], name)
            parts[name] = (
                paired_mean_difference(a, b, pat, **boot) | {"n_stays": int(len(a))} if len(a) else None
            )
        usable = {k: v for k, v in parts.items() if v}
        if {"driver_precision", "citation_precision", "driver_recall_at_3"} <= set(usable):
            precisions = [usable["driver_precision"], usable["citation_precision"]]
            tests["H5"] = {
                "parts": usable,
                "p_value": max(p["p_value"] for p in precisions),
                "both_precisions_higher": all(p["difference"] > 0 for p in precisions),
                "recall_non_inferior": usable["driver_recall_at_3"]["low"] > -0.02,
                "metric": "E8 - E7: driver and citation precision (intersection); recall non-inferiority",
            }
        else:
            not_run["H5"] = "citation or driver metrics are undefined for the E8 / E7 stays"
    else:
        not_run["H5"] = "DPO was not run (no E8 texts)"

    holm = holm_bonferroni({h: t["p_value"] for h, t in tests.items()}, alpha=config.alpha) if tests else {}
    for h, t in tests.items():
        t["holm"] = holm[h]
        sign = t.get("expected_sign")
        if h == "H5":
            right = t["both_precisions_higher"] and t["recall_non_inferior"]
        elif h == "H4":
            right = t["higher_than_both"]
        else:
            right = (t["difference"] > 0) if sign == "positive" else (t["difference"] < 0)
        t["supported"] = bool(holm[h]["reject"] and right)
    out["hypotheses"] = {"tests": tests, "not_run": not_run}

    runnable, lacking = runnable_rows(
        arms=bundle.arms.keys(),
        packets={"attribution", "shap", "summary", "random"},
        texts=set(bundle.explained),
        has_passages=any(it.passages for rows in bundle.explained.values() for it in rows),
    )
    out["grid"] = {"runnable": [r.id for r in runnable], "lacking": lacking}
    return out


# ---- report ------------------------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.4f}"
    return str(x)


def _write_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([[_fmt(c) for c in r] for r in rows])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    results: dict[str, Any], out_dir: Path, config: EvalConfig, *, manifest: dict | None = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    rows = []
    for arm, d in results["predictive"].items():
        for metric, v in d["all_test_stays"].items():
            rows.append([arm, metric, v["value"], v["low"], v["high"], d["n_stays"]])
    _write_table(tables / "predictive", ["arm", "metric", "value", "low", "high", "n_stays"], rows)

    rows = [
        [rid, m, v["mean"], v["low"], v["high"], v["n"], v["missing"]]
        for rid, d in results["explanation"].items()
        for m, v in d.items()
    ]
    _write_table(
        tables / "explanation_faithfulness", ["row", "metric", "mean", "low", "high", "n", "undefined"], rows
    )

    rows = [
        [rid, m, v["ratio"], v["low"], v["high"]]
        for rid, d in results["grounding"].items()
        for m, v in d.items()
        if m != "counts"
    ]
    _write_table(tables / "groundedness", ["row", "metric", "ratio", "low", "high"], rows)

    rows = [
        [h, t["metric"], t.get("difference"), t.get("low"), t.get("high"), t["p_value"],
         t["holm"]["p_adjusted"], t["supported"]]
        for h, t in results["hypotheses"]["tests"].items()
    ]  # fmt: skip
    rows += [
        [h, why, None, None, None, None, None, "not run"]
        for h, why in results["hypotheses"]["not_run"].items()
    ]
    _write_table(
        tables / "hypotheses",
        ["hypothesis", "metric", "difference", "low", "high", "p", "p_holm", "supported"],
        rows,
    )

    from fate_icu.eval.grid import grid_table

    grid = grid_table()
    _write_table(tables / "grid", list(grid[0]), [list(r.values()) for r in grid])

    from fate_icu.utils.manifest import write_run_manifest

    write_run_manifest(
        out_dir,
        run_id=out_dir.name,
        config=config.model_dump(),
        seeds=[config.seed],
        **(manifest or {}),
    )
    if config.figures:
        plot_explanation(results, out_dir)
    return out_dir


def plot_explanation(results: dict[str, Any], out_dir: Path) -> Path | None:
    """A bar chart of the unattributed-driver rate per row with its interval (matplotlib, optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    ids = [r for r, d in results["explanation"].items() if "unattributed_rate" in d]
    if not ids:
        return None
    means = [results["explanation"][r]["unattributed_rate"]["mean"] for r in ids]
    lows = [
        m - results["explanation"][r]["unattributed_rate"]["low"] for r, m in zip(ids, means, strict=True)
    ]
    highs = [
        results["explanation"][r]["unattributed_rate"]["high"] - m for r, m in zip(ids, means, strict=True)
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(ids, means, yerr=[lows, highs], capsize=3)
    ax.set_ylabel("unattributed-driver rate")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "unattributed_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_checker(config: EvalConfig) -> G.Checker:
    if config.checker == "nli":
        return G.NLIChecker(config.nli_model, revision=config.nli_revision)
    return G.LexicalChecker()


def run(config: EvalConfig, out_root: Path | None = None) -> Path:
    if config.source != "synthetic":
        raise SystemExit(
            "the full evaluation reads the Stage 3, 4, 5 and 6 outputs on the GCP VM; that loader is added "
            "when Stage 6 has generated texts to evaluate (docs/lab_runbook.md, Stage 7). "
            "Run the smoke config here."
        )
    from fate_icu.eval.smoke_data import synthetic_bundle
    from fate_icu.utils.manifest import new_run_id

    if out_root is None:
        from fate_icu.utils.env import load_settings

        out_root = Path(config.reports_dir) if config.reports_dir else load_settings().reports_dir
    bundle = synthetic_bundle(config.n_stays, config.seed)
    lexicon = X.build_lexicon(concepts=["Effusion", "Edema", "Atelectasis"])
    results = evaluate(bundle, config, make_checker(config), lexicon)
    return write_report(results, Path(out_root) / new_run_id(config.run_name), config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = run(config)
    results = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "report": str(out),
        "grid_runnable": results["grid"]["runnable"],
        "hypotheses": {
            h: {"p_holm": t["holm"]["p_adjusted"], "supported": t["supported"]}
            for h, t in results["hypotheses"]["tests"].items()
        },
        "not_run": results["hypotheses"]["not_run"],
    }
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
