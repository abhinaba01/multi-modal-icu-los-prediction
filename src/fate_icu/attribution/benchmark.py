"""Synthetic ground-truth benchmark for the attribution methods (Stage 4).

    python -m fate_icu.attribution.benchmark --config configs/attribution/benchmark.yaml

Trains a small model on stays whose label depends on four planted evidence units
(``attribution/synthetic.py``), then scores each method on stays from the test split where at
least one planted unit is active:

* **recovery**: precision and recall at k = 1, 3, 5 of the planted units, per modality;
* **faithfulness**: deletion AOPC, comprehensiveness and sufficiency at k = 1, 3, 5 (DeYoung et
  al., 2020), against a random ordering of the same units;
* **driver selection**: precision and recall of the final drivers after the two-step rule;
* **model randomisation** (Adebayo et al., 2018): rank agreement with the same method on a
  model whose weights were re-drawn.

These are real results (the data are synthetic, so nothing here is credentialed). The numbers
decide the default method recorded in docs/decisions.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fate_icu.attribution import methods as M
from fate_icu.attribution.gates import GatedLogit, stay_tensors
from fate_icu.attribution.select import SelectionConfig, candidate_units, fit_tau, select_drivers
from fate_icu.attribution.synthetic import Planted, planted_stays
from fate_icu.attribution.units import StayScorer
from fate_icu.eval.attr_faithfulness import (
    K_VALUES,
    comprehensiveness_sufficiency,
    curves,
    random_order,
    randomisation_agreement,
    ranking,
)
from fate_icu.eval.attr_faithfulness import precision_recall_at_k as prk
from fate_icu.models.fusion import FusionNet, ModelSpec
from fate_icu.models.train import TrainConfig, predict, train_model
from fate_icu.utils.env import REPO_ROOT, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import seed_everything

MODALITY_METHODS = {
    "timeseries": M.TS,
    "static": M.STATIC,
    "image": M.CONCEPT,
}


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = "attribution_benchmark"
    seed: int = 666
    n_stays: int = 700
    n_eval: int = 60  # test stays explained (long stays with a planted unit)
    model: dict = Field(default_factory=lambda: {"d": 32, "heads": 4, "layers": 1, "dropout": 0.1})
    epochs: int = 60
    lr: float = 0.003
    ig_steps: int = 32
    mask_epochs: int = 500
    mask_lr: float = 0.01
    mask_lambda_1: float = 3.0
    dynamask_epochs: int = 150
    ts_methods: list[str] = Field(default_factory=lambda: list(M.TS_METHODS))
    static_methods: list[str] = Field(default_factory=lambda: list(M.STATIC_METHODS))
    image_methods: list[str] = Field(default_factory=lambda: list(M.IMAGE_METHODS))
    combos: list[list[str]] = Field(
        default_factory=lambda: [
            ["extremal_mask", "ig", "ig"],
            ["occlusion", "ablation", "ablation"],
            ["ig", "ig", "ig"],
        ]
    )
    n_boot: int = 200
    min_auroc: float = 0.9  # the model must have learned the planted rule before it is explained


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return BenchmarkConfig.model_validate(yaml.safe_load(handle))


def method_config(cfg: BenchmarkConfig) -> M.MethodConfig:
    return M.MethodConfig(
        ig_steps=cfg.ig_steps,
        mask_epochs=cfg.mask_epochs,
        mask_lr=cfg.mask_lr,
        mask_lambda_1=cfg.mask_lambda_1,
        dynamask_epochs=cfg.dynamask_epochs,
        seed=cfg.seed,
    )


def train_planted_model(planted: Planted, cfg: BenchmarkConfig):
    spec = ModelSpec(("static", "ts", "image"), "attention", **cfg.model)
    trained = train_model(
        spec,
        planted.data,
        TrainConfig(epochs=cfg.epochs, batch_size=32, lr=cfg.lr, patience=cfg.epochs),
        cfg.seed,
    )
    test = planted.data.rows("test")
    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(planted.data.y_long[test], predict(trained, planted.data, test)["p_long"]))
    if auroc < cfg.min_auroc:
        raise RuntimeError(
            f"the benchmark model reached test AUROC {auroc:.2f}, below the {cfg.min_auroc:.2f} it needs "
            "to count as having learned the planted rule"
        )
    return trained, auroc


def _interval(values: list[float], n_boot: int, seed: int) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, size=len(v)).mean() for _ in range(n_boot)]
    return {
        "mean": float(v.mean()),
        "low": float(np.quantile(boots, 0.025)),
        "high": float(np.quantile(boots, 0.975)),
        "n": int(len(v)),
    }


def modality_units(planted: Planted, modality: str) -> np.ndarray:
    space = planted.space
    if modality == "timeseries":
        return np.arange(space.n_ts)
    if modality == "static":
        return np.arange(space.static_slice.start, space.static_slice.stop)
    return np.arange(space.concept_slice.start, space.concept_slice.stop)


def evaluate_modality(model, planted, rows, fixed, modality, name, fn, cfg, mcfg):
    """Recovery and faithfulness of one method on the stays ``rows`` (those with truth in it)."""
    space = planted.space
    units = modality_units(planted, modality)
    scores_all = fn(model, space, fixed, mcfg)  # [N, len(units)]
    out: dict[str, list[float]] = {}
    rng = np.random.default_rng(cfg.seed)
    for i, row in enumerate(rows):
        scores = np.zeros(len(space), dtype=np.float32)
        scores[units] = scores_all[i]
        usable = np.zeros(len(space), dtype=bool)
        usable[units] = True
        truth = planted.truth[row] & usable
        scorer = StayScorer(model, space, planted.data, int(row))
        order = ranking(scores, usable)
        random = random_order(len(space), usable, rng)
        for k in K_VALUES:
            p, r = prk(np.where(usable, scores, 0.0), truth, k)
            out.setdefault(f"precision@{k}", []).append(p)
            out.setdefault(f"recall@{k}", []).append(r)
        faithful = comprehensiveness_sufficiency(order, len(space), scorer.logits)
        baseline = comprehensiveness_sufficiency(random, len(space), scorer.logits)
        for k in K_VALUES:
            out.setdefault(f"comprehensiveness@{k}", []).append(faithful[k]["comprehensiveness"])
            out.setdefault(f"sufficiency@{k}", []).append(faithful[k]["sufficiency"])
            out.setdefault(f"random_comprehensiveness@{k}", []).append(baseline[k]["comprehensiveness"])
            out.setdefault(f"random_sufficiency@{k}", []).append(baseline[k]["sufficiency"])
        curve = curves(order, len(space), scorer.logits, max_steps=5)
        out.setdefault("deletion_aopc@5", []).append(curve["aopc"])
        out.setdefault("random_deletion_aopc@5", []).append(
            curves(random, len(space), scorer.logits, max_steps=5)["aopc"]
        )
    return {key: _interval(v, cfg.n_boot, cfg.seed) for key, v in out.items()}, scores_all


def evaluate_selection(model, planted, rows, fixed, combo, cfg, mcfg, tau, sel):
    """Precision and recall of the final drivers for one (time series, static, image) combo."""
    space = planted.space
    scores = M.attribute(model, space, fixed, mcfg, ts=combo[0], static=combo[1], concept=combo[2])
    precision, recall, counts, top1, mass = [], [], [], [], []
    for i, row in enumerate(rows):
        scorer = StayScorer(model, space, planted.data, int(row))
        drivers = select_drivers(scorer, scores[i], tau, sel)
        truth = set(np.flatnonzero(planted.truth[row]).tolist())
        covered = {u for d in drivers for u in d.units}
        hits = [bool(set(d.units) & truth) for d in drivers]
        precision.append(float(np.mean(hits)) if drivers else 0.0)
        recall.append(len(truth & covered) / len(truth))
        counts.append(float(len(drivers)))
        top1.append(float(hits[0]) if drivers else 0.0)  # drivers are sorted by |delta|
        total = sum(abs(d.delta) for d in drivers)
        mass.append(
            sum(abs(d.delta) for d, h in zip(drivers, hits, strict=True) if h) / total if total else 0.0
        )
    return {
        "driver_precision": _interval(precision, cfg.n_boot, cfg.seed),
        "driver_recall": _interval(recall, cfg.n_boot, cfg.seed),
        "top1_is_planted": _interval(top1, cfg.n_boot, cfg.seed),
        "effect_on_planted": _interval(mass, cfg.n_boot, cfg.seed),
        "n_drivers": _interval(counts, cfg.n_boot, cfg.seed),
    }, scores


def run(cfg: BenchmarkConfig) -> dict:
    seed_everything(cfg.seed)
    settings = load_settings()
    run_id = new_run_id(cfg.run_name)
    planted = planted_stays(n=cfg.n_stays, seed=cfg.seed)
    trained, auroc = train_planted_model(planted, cfg)
    layout, space, data = planted.layout, planted.space, planted.data
    model = GatedLogit(trained.model, layout)
    mcfg = method_config(cfg)
    sel = SelectionConfig()

    test = data.rows("test")
    probs = predict(trained, data, test)["p_long"]
    chosen = test[(data.y_long[test] > 0.5) & (probs > 0.5) & planted.truth[test].any(axis=1)][: cfg.n_eval]
    if len(chosen) < 10:
        raise RuntimeError("fewer than 10 long test stays with a planted unit: raise n_stays")
    fixed = stay_tensors(data, chosen)

    methods = {"timeseries": cfg.ts_methods, "static": cfg.static_methods, "image": cfg.image_methods}
    results: dict = {"test_auroc": auroc, "n_explained": int(len(chosen)), "modalities": {}}
    for modality, names in methods.items():
        have = np.array([planted.truth[r][modality_units(planted, modality)].any() for r in chosen])
        if not have.any():
            continue
        sub = chosen[have]
        sub_fixed = stay_tensors(data, sub)
        results["modalities"][modality] = {"n_stays": int(len(sub)), "methods": {}}
        for name in names:
            entry, _ = evaluate_modality(
                model, planted, sub, sub_fixed, modality, name, MODALITY_METHODS[modality][name], cfg, mcfg
            )
            results["modalities"][modality]["methods"][name] = entry

    # tau on validation stays: random non-candidate units of the default combination
    val = data.rows("val")[:40]
    val_scores = M.attribute(
        model, space, stay_tensors(data, val), mcfg, ts="occlusion", static="ablation", concept="ablation"
    )
    scorers = [StayScorer(model, space, data, int(r)) for r in val]
    candidates = [candidate_units(val_scores[i], space, sel.per_modality) for i in range(len(val))]
    tau = fit_tau(scorers, candidates, sel, cfg.seed)
    results["tau"] = tau
    results["selection"] = {}
    all_scores = {}
    for combo in cfg.combos:
        key = "+".join(combo)
        results["selection"][key], all_scores[key] = evaluate_selection(
            model, planted, chosen, fixed, combo, cfg, mcfg, tau, sel
        )

    # model randomisation: the same methods on a model with re-drawn weights
    torch.manual_seed(cfg.seed + 1)
    random_net = FusionNet(trained.spec, data.dims).eval()
    random_model = GatedLogit(random_net, layout)
    results["randomisation_spearman"] = {}
    for combo in cfg.combos:
        key = "+".join(combo)
        random_scores = M.attribute(
            random_model, space, fixed, mcfg, ts=combo[0], static=combo[1], concept=combo[2]
        )
        agreement = [
            randomisation_agreement(all_scores[key][i], random_scores[i]) for i in range(len(chosen))
        ]
        results["randomisation_spearman"][key] = _interval(agreement, cfg.n_boot, cfg.seed)

    out_dir = settings.outputs_dir / "attribution" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    write_run_manifest(
        out_dir,
        run_id=run_id,
        config=cfg.model_dump(),
        seeds=[cfg.seed],
        data={"data": "synthetic planted benchmark"},
    )
    results["out_dir"] = str(out_dir)
    return results


def _cell(m: dict, width: int = 18) -> str:
    """``mean (low-high)``: the mean over stays with its 95 % bootstrap interval."""
    return f"{m['mean']:.2f} ({m['low']:.2f}-{m['high']:.2f})".rjust(width)


def summary_table(results: dict) -> str:
    """A plain-text table: means over the explained stays with 95 % intervals (k = 3 for time
    series, k = 1 for the small modalities, where three units would be everything)."""
    head = f"model test AUROC {results['test_auroc']:.3f}, {results['n_explained']} stays explained"
    lines = [f"{head}, tau {results['tau']:.3f}", ""]
    columns = ["precision@1", "recall@3", "comprehens.", "(random)", "sufficiency", "(random)", "AOPC@5"]
    lines.append(f"{'modality':11s} {'method':14s} {'k':>2s} " + " ".join(f"{c:>18s}" for c in columns))
    for modality, block in results["modalities"].items():
        k = 3 if modality == "timeseries" else 1
        for name, m in block["methods"].items():
            cells = [
                m["precision@1"],
                m["recall@3"],
                m[f"comprehensiveness@{k}"],
                m[f"random_comprehensiveness@{k}"],
                m[f"sufficiency@{k}"],
                m[f"random_sufficiency@{k}"],
                m["deletion_aopc@5"],
            ]
            lines.append(f"{modality:11s} {name:14s} {k:2d} " + " ".join(_cell(c) for c in cells))
    columns = ["precision", "recall", "top-1 planted", "effect on planted", "drivers", "random-model agr."]
    lines += ["", f"{'combination':30s} " + " ".join(f"{c:>18s}" for c in columns)]
    for key, m in results["selection"].items():
        cells = [
            m["driver_precision"],
            m["driver_recall"],
            m["top1_is_planted"],
            m["effect_on_planted"],
            m["n_drivers"],
            results["randomisation_spearman"][key],
        ]
        lines.append(f"{key:30s} " + " ".join(_cell(c) for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    results = run(load_config(path if path.is_absolute() else REPO_ROOT / path))
    print(summary_table(results))
    print(f"\nwritten to {results['out_dir']}")


if __name__ == "__main__":
    main()
