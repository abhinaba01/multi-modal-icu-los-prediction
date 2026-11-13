"""Retrieval evaluation (Stage 5): pooled relevance grades, precision@3 and nDCG@5, paired tests.

Protocol (docs/decisions.md; grading rules in docs/rag_grading_guide.md):

* queries: the 50 "longer" queries, one per driver key;
* pool: for each query, the union of the top 5 passages of the four systems (bm25, dense,
  hybrid, hybrid_rerank), shown in a blind order (shuffled by a hash, no system names);
* grades: 0 = off-topic, 1 = related background, 2 = directly relevant;
* per query: P@3 with relevant = grade >= 1 (primary) and grade = 2 (strict), and nDCG@5
  with gains 2^grade - 1, discount log2(rank + 1), and the ideal ordering of every judged
  passage of that query (0 when none is relevant; such queries are counted);
* per system: the mean over queries and a percentile bootstrap 95% CI (queries resampled);
* comparisons (hybrid vs bm25, hybrid vs dense, hybrid_rerank vs hybrid, dense vs bm25):
  mean difference, paired bootstrap 95% CI, two-sided paired sign-flip permutation p
  (exact when 2^n <= n_perm), Holm-Bonferroni adjusted within each metric.

Every system's top 5 is in the pool, so every metric is fully judged; an unjudged passage
(labels from another index) is an error. The committed ``configs/rag/relevance_labels.tsv``
holds ids and grades only, never passage text.

    python -m fate_icu.rag.evaluate sheet  --config configs/rag/full.yaml
    python -m fate_icu.rag.label           --config configs/rag/full.yaml   # you grade
    python -m fate_icu.rag.evaluate import --config configs/rag/full.yaml --rater <name>
    python -m fate_icu.rag.evaluate score  --config configs/rag/full.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank")
COMPARISONS = (("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid_rerank", "hybrid"), ("dense", "bm25"))
GRADES = (0, 1, 2)
SHEET_VERSION = "1"
SHEET_COLUMNS = [
    "sheet_version",
    "index_version",
    "query_id",
    "query",
    "item",
    "passage_id",
    "source",
    "title",
    "section",
    "locator",
    "text",
    "grade",
    "note",
]
LABEL_COLUMNS = ["query_id", "passage_id", "grade", "rater", "index_version"]
SHEET_FILE = "labelling_sheet.tsv"


# --- pool and sheet --------------------------------------------------------------------


def build_pool(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    query_ids: Sequence[str],
    systems: Sequence[str] = SYSTEMS,
    depth: int = 5,
) -> dict[str, list[str]]:
    """The union of every system's top ``depth`` per query, in order of first appearance."""
    pool = {}
    for query_id in query_ids:
        if query_id not in rankings:
            raise KeyError(f"no rankings for query {query_id}")
        seen: list[str] = []
        for system in systems:
            if system not in rankings[query_id]:
                raise KeyError(f"query {query_id} has no {system} ranking")
            for pid in rankings[query_id][system][:depth]:
                if pid not in seen:
                    seen.append(pid)
        pool[query_id] = seen
    return pool


def blind_order(pool: Mapping[str, Sequence[str]], seed: int) -> dict[str, list[str]]:
    """A deterministic shuffle within each query that hides the system and the rank."""

    def key(qid: str, pid: str) -> str:
        return hashlib.sha256(f"{seed}:{qid}:{pid}".encode()).hexdigest()

    return {qid: sorted(pids, key=lambda pid: key(qid, pid)) for qid, pids in pool.items()}


@dataclass(frozen=True)
class SheetRow:
    sheet_version: str
    index_version: str
    query_id: str
    query: str
    item: int
    passage_id: str
    source: str
    title: str
    section: str
    locator: str
    text: str
    grade: int | None
    note: str = ""


def _cell(value: Any) -> str:
    return " ".join(str(value if value is not None else "").replace("\t", " ").split())


def write_sheet_rows(path: str | Path, rows: Sequence[SheetRow]) -> None:
    """Write sheet rows as tab-separated UTF-8 with a BOM (Excel opens it), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\r\n")
    writer.writerow(SHEET_COLUMNS)
    for row in rows:
        values = [getattr(row, column) for column in SHEET_COLUMNS]
        writer.writerow(["" if v is None else _cell(v) for v in values])
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    os.replace(tmp, path)


def write_labelling_sheet(
    path: str | Path,
    pool_order: Mapping[str, Sequence[str]],
    queries: Mapping[str, str],
    passages: Mapping[str, Any],
    index_version: str,
) -> int:
    """The blind sheet to grade: one row per (query, pooled passage). Returns the row count."""
    rows = []
    for query_id, pids in pool_order.items():
        for item, pid in enumerate(pids, start=1):
            p = passages[pid]
            rows.append(
                SheetRow(
                    SHEET_VERSION,
                    index_version,
                    query_id,
                    queries[query_id],
                    item,
                    pid,
                    p.source,
                    p.title,
                    p.section,
                    p.locator,
                    p.text,
                    None,
                )
            )
    write_sheet_rows(path, rows)
    return len(rows)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")  # Excel "Unicode Text"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    return raw.decode("utf-8")


def parse_grade(value: str, where: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    for grade in GRADES:
        if text in (str(grade), f"{grade}.0"):
            return grade
    raise ValueError(f"{where}: grade {value!r} is not 0, 1 or 2")


def read_labelling_sheet(path: str | Path) -> list[SheetRow]:
    """Rows of a sheet saved by this module, by Excel (UTF-8 or Unicode Text) or by label.py."""
    path = Path(path)
    reader = csv.reader(io.StringIO(_read_text(path), newline=""), delimiter="\t")
    header = next(reader, None)
    if header != SHEET_COLUMNS:
        raise ValueError(f"{path.name}: unexpected columns {header}")
    rows = []
    for number, values in enumerate(reader, start=2):
        if not any(v.strip() for v in values):
            continue
        if len(values) != len(SHEET_COLUMNS):
            raise ValueError(f"{path.name} line {number}: {len(values)} cells, expected {len(SHEET_COLUMNS)}")
        record = dict(zip(SHEET_COLUMNS, values, strict=True))
        rows.append(
            SheetRow(
                **{
                    **record,
                    "item": int(record["item"]),
                    "grade": parse_grade(record["grade"], f"line {number}"),
                }
            )
        )
    return rows


# --- labels ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    query_id: str
    passage_id: str
    grade: int
    rater: str
    index_version: str


def export_labels(rows: Sequence[SheetRow], rater: str, *, allow_partial: bool = False) -> list[Label]:
    """The graded rows as labels (ids and grades only).

    Raises:
        ValueError: Rows are ungraded (unless ``allow_partial``), or the rater name is empty.
    """
    if not rater.strip():
        raise ValueError("give a rater name")
    ungraded = sum(r.grade is None for r in rows)
    if ungraded and not allow_partial:
        raise ValueError(f"{ungraded} of {len(rows)} rows are not graded yet")
    return [
        Label(r.query_id, r.passage_id, r.grade, rater.strip(), r.index_version)
        for r in rows
        if r.grade is not None
    ]


def read_labels(path: str | Path) -> list[Label]:
    path = Path(path)
    if not path.exists():
        return []
    reader = csv.reader(io.StringIO(_read_text(path), newline=""), delimiter="\t")
    header = next(reader, None)
    if header != LABEL_COLUMNS:
        raise ValueError(f"{path.name}: unexpected columns {header}")
    labels = []
    for number, values in enumerate(reader, start=2):
        if values:
            query_id, passage_id, grade, rater, index_version = values
            labels.append(
                Label(query_id, passage_id, parse_grade(grade, f"line {number}"), rater, index_version)
            )
    return labels


def write_labels(path: str | Path, labels: Sequence[Label]) -> int:
    """Replace the given raters' rows in the labels file, keep other raters'. Returns the row count."""
    path = Path(path)
    raters = {label.rater for label in labels}
    rows = [label for label in read_labels(path) if label.rater not in raters] + list(labels)
    keys = [(r.rater, r.query_id, r.passage_id) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("a (rater, query, passage) appears twice")
    rows.sort(key=lambda r: (r.rater, r.query_id, r.passage_id))
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    writer.writerows([r.query_id, r.passage_id, r.grade, r.rater, r.index_version] for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return len(rows)


def grades_of(labels: Sequence[Label], rater: str | None = None) -> dict[tuple[str, str], int]:
    """``{(query_id, passage_id): grade}`` for one rater (the only one when ``rater`` is None)."""
    raters = sorted({label.rater for label in labels})
    if rater is None:
        if len(raters) != 1:
            raise ValueError(f"the labels hold raters {raters}: choose one")
        rater = raters[0]
    return {(label.query_id, label.passage_id): label.grade for label in labels if label.rater == rater}


# --- metrics ---------------------------------------------------------------------------


def _grade(pid: str, grades: Mapping[str, int], unjudged: str) -> int:
    if pid in grades:
        return grades[pid]
    if unjudged == "zero":
        return 0
    raise ValueError(f"passage {pid} is in a system's top k but has no grade (labels from another index?)")


def precision_at_k(
    ranked: Sequence[str], grades: Mapping[str, int], k: int, threshold: int, *, unjudged: str = "error"
) -> float:
    """Share of the first ``k`` results with grade >= ``threshold``; the denominator is ``k``."""
    return sum(_grade(pid, grades, unjudged) >= threshold for pid in ranked[:k]) / k


def dcg(gains: Sequence[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int, *, unjudged: str = "error") -> float:
    """nDCG@k with the ideal ordering of every judged passage of the query (0 if none is relevant)."""
    ideal = dcg(sorted(grades.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg([_grade(pid, grades, unjudged) for pid in ranked[:k]], k) / ideal


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling the values (queries) with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p(differences: Sequence[float], n_perm: int, seed: int) -> float:
    """Two-sided paired sign-flip permutation p-value of the mean difference."""
    diff = np.asarray(differences, dtype=float)
    n = len(diff)
    if n == 0 or not diff.any():
        return 1.0
    observed = abs(diff.mean()) - 1e-12
    if 2**n <= n_perm:  # every sign pattern
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs((signs * diff).mean(axis=1)) >= observed))
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    extreme = int(np.sum(np.abs((signs * diff).mean(axis=1)) >= observed))
    return (extreme + 1) / (n_perm + 1)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m, running, adjusted = len(ordered), 0.0, {}
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = GRADES) -> float:
    """Cohen's kappa with quadratic weights (agreement between two raters on the same items)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty grade lists")
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    denominator = (weights * expected).sum()
    return 1.0 if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def metric_names(k_precision: int = 3, k_ndcg: int = 5) -> list[str]:
    return [f"P@{k_precision}", f"P@{k_precision} strict", f"nDCG@{k_ndcg}"]


def evaluate_systems(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    grades: Mapping[tuple[str, str], int],
    query_ids: Sequence[str],
    systems: Sequence[str] = SYSTEMS,
    *,
    k_precision: int = 3,
    k_ndcg: int = 5,
    n_boot: int = 1000,
    n_perm: int = 10_000,
    seed: int = 666,
    unjudged: str = "error",
) -> dict[str, Any]:
    """Per-query metrics, per-system means with CIs, and paired comparisons (see the module rules)."""
    if not query_ids:
        raise ValueError("no queries to evaluate")
    by_query: dict[str, dict[str, int]] = {qid: {} for qid in query_ids}
    for (qid, pid), grade in grades.items():
        if qid in by_query:
            by_query[qid][pid] = grade
    names = metric_names(k_precision, k_ndcg)
    per_query: dict[str, dict[str, list[float]]] = {}
    for system in systems:
        values = {name: [] for name in names}
        for qid in query_ids:
            ranked = list(rankings[qid][system])
            judged = by_query[qid]
            values[names[0]].append(precision_at_k(ranked, judged, k_precision, 1, unjudged=unjudged))
            values[names[1]].append(precision_at_k(ranked, judged, k_precision, 2, unjudged=unjudged))
            values[names[2]].append(ndcg_at_k(ranked, judged, k_ndcg, unjudged=unjudged))
        per_query[system] = values
    summary = {
        system: {
            name: {"mean": float(np.mean(v)), "ci95": list(bootstrap_ci(v, n_boot, seed))}
            for name, v in per_query[system].items()
        }
        for system in systems
    }
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        rows, pvalues = [], {}
        for a, b in COMPARISONS:
            if a not in systems or b not in systems:
                continue
            diff = np.asarray(per_query[a][name]) - np.asarray(per_query[b][name])
            label = f"{a} vs {b}"
            pvalues[label] = sign_flip_p(diff, n_perm, seed)
            rows.append(
                {
                    "comparison": label,
                    "difference": float(diff.mean()),
                    "ci95": list(bootstrap_ci(diff, n_boot, seed)),
                    "p": pvalues[label],
                }
            )
        adjusted = holm(pvalues)
        comparisons[name] = [{**row, "p_holm": adjusted[row["comparison"]]} for row in rows]
    return {
        "queries": len(query_ids),
        "systems": list(systems),
        "metrics": names,
        "queries_without_relevant_passage": sum(
            not any(g >= 1 for g in by_query[q].values()) for q in query_ids
        ),
        "judged_passages": sum(len(v) for v in by_query.values()),
        "summary": summary,
        "comparisons": comparisons,
        "per_query": {s: {name: list(v) for name, v in m.items()} for s, m in per_query.items()},
        "settings": {
            "k_precision": k_precision,
            "k_ndcg": k_ndcg,
            "n_boot": n_boot,
            "n_perm": n_perm,
            "seed": seed,
        },
    }


def format_markdown(result: Mapping[str, Any]) -> str:
    names = result["metrics"]
    lines = [
        "| System | " + " | ".join(f"{n} (95% CI)" for n in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for system in result["systems"]:
        cells = []
        for name in names:
            m = result["summary"][system][name]
            cells.append(f"{m['mean']:.3f} ({m['ci95'][0]:.3f}-{m['ci95'][1]:.3f})")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += ["", "| Metric | Comparison | Difference (95% CI) | p | Holm p |", "|---|---|---|---|---|"]
    for name in names:
        for row in result["comparisons"][name]:
            lines.append(
                f"| {name} | {row['comparison']} | {row['difference']:+.3f} ({row['ci95'][0]:+.3f} to "
                f"{row['ci95'][1]:+.3f}) | {row['p']:.4f} | {row['p_holm']:.4f} |"
            )
    lines.append("")
    lines.append(
        f"{result['queries']} queries, {result['judged_passages']} judged passages; "
        f"{result['queries_without_relevant_passage']} queries without a relevant passage."
    )
    return "\n".join(lines) + "\n"


# --- the query bank ---------------------------------------------------------------------


def rankings_from_bank(bank: Any) -> dict[str, dict[str, list[str]]]:
    """``{query_id: {system: [passage_id, ...]}}`` from a QueryBank."""
    return {
        qid: {s: [h.passage_id for h in q.rankings[s]] for s in SYSTEMS} for qid, q in bank.queries.items()
    }


def evaluation_queries(bank: Any, direction: str) -> list[str]:
    return sorted(qid for qid, q in bank.queries.items() if q.direction == direction)


def make_sheet_from_bank(
    bank_path: str | Path,
    sheet_path: str | Path,
    *,
    direction: str = "longer",
    depth: int = 5,
    seed: int = 666,
) -> dict[str, Any]:
    """Write the blind labelling sheet for the bank's ``direction`` queries."""
    from fate_icu.rag.retrieve import load_query_bank

    bank = load_query_bank(bank_path)
    query_ids = evaluation_queries(bank, direction)
    pool = build_pool(rankings_from_bank(bank), query_ids, depth=depth)
    missing = {pid for pids in pool.values() for pid in pids} - set(bank.passages)
    if missing:
        raise RuntimeError(
            f"{len(missing)} pooled passages are not stored in the bank (store_depth < {depth}?)"
        )
    rows = write_labelling_sheet(
        sheet_path,
        blind_order(pool, seed),
        {qid: bank.queries[qid].text for qid in query_ids},
        bank.passages,
        bank.index_version,
    )
    return {
        "sheet": str(sheet_path),
        "queries": len(query_ids),
        "rows": rows,
        "index_version": bank.index_version,
    }


def score_bank(
    bank_path: str | Path,
    labels_path: str | Path,
    out_dir: str | Path,
    *,
    direction: str = "longer",
    rater: str | None = None,
    n_boot: int = 1000,
    n_perm: int = 10_000,
    seed: int = 666,
) -> dict[str, Any]:
    """Score the four systems against the labels; write metrics.json and metrics.md."""
    from fate_icu.rag.retrieve import load_query_bank

    bank = load_query_bank(bank_path)
    labels = read_labels(labels_path)
    if not labels:
        raise FileNotFoundError(f"no relevance labels in {labels_path}")
    chosen = rater or sorted({label.rater for label in labels})[0]
    versions = {label.index_version for label in labels if label.rater == chosen}
    if versions != {bank.index_version}:
        raise RuntimeError(
            f"the labels of {chosen} are for index {sorted(versions)}, the bank is {bank.index_version}"
        )
    result = evaluate_systems(
        rankings_from_bank(bank),
        grades_of(labels, chosen),
        evaluation_queries(bank, direction),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    result.update({"index_version": bank.index_version, "rater": chosen, "direction": direction})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def _config_paths(config_path: str) -> dict[str, Path]:
    from fate_icu.rag.config import load_config, rag_paths, repo_path
    from fate_icu.utils.env import load_settings

    config = load_config(config_path)
    paths = rag_paths(config, load_settings())
    return {
        "bank": paths.index(paths.current("index")) / "query_bank.json",
        "sheet": paths.eval / SHEET_FILE,
        "labels": repo_path(config.evaluation.labels),
        "out": paths.eval,
        "config": config,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sheet", "import", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", help="configs/rag/full.yaml (gives every path below)")
        sub.add_argument("--bank")
        sub.add_argument("--sheet")
        sub.add_argument("--labels")
        sub.add_argument("--out")
        sub.add_argument("--rater")
        sub.add_argument("--direction", default=None)
    args = parser.parse_args(argv)
    paths = _config_paths(args.config) if args.config else {}
    config = paths.get("config")
    direction = args.direction or (config.evaluation.direction if config else "longer")

    def path(name: str) -> Path:
        value = getattr(args, name) or paths.get(name)
        if value is None:
            parser.error(f"give --{name} or --config")
        return Path(value)

    if args.command == "sheet":
        depth = config.evaluation.pool_depth if config else 5
        seed = config.evaluation.seed if config else 666
        report = make_sheet_from_bank(
            path("bank"), path("sheet"), direction=direction, depth=depth, seed=seed
        )
    elif args.command == "import":
        if not args.rater:
            parser.error("give --rater (your name or initials)")
        rows = read_labelling_sheet(path("sheet"))
        report = {
            "labels_written": write_labels(path("labels"), export_labels(rows, args.rater)),
            "file": str(path("labels")),
        }
    else:
        evaluation = config.evaluation if config else None
        result = score_bank(
            path("bank"),
            path("labels"),
            path("out"),
            direction=direction,
            rater=args.rater,
            n_boot=evaluation.n_boot if evaluation else 1000,
            n_perm=evaluation.n_perm if evaluation else 10_000,
            seed=evaluation.seed if evaluation else 666,
        )
        print(format_markdown(result))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
