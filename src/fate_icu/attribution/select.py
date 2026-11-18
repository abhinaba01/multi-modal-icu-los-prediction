"""Driver selection (Stage 4): two steps, because scores of different methods do not compare.

1. Within each modality keep the top ``per_modality`` units by that modality's method (by the
   size of the score, either sign).
2. Across modalities score every candidate by Δ = logit(x) - logit(x without the unit), the
   in-distribution removal of ``gates.py``. Keep |Δ| >= tau, merge adjacent blocks of the same
   variable that pushed the same way, recompute Δ for the merged unit and apply tau again, then
   keep at most ``max_drivers``, sorted by |Δ|. Δ > 0 pushed toward a longer stay.

``tau`` is the ``quantile`` (0.95) of |Δ| over random units that were *not* candidates, taken on
validation stays only, so a driver must matter more than a typical unit does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fate_icu.attribution.units import StayScorer, UnitSpace


@dataclass(frozen=True)
class SelectionConfig:
    per_modality: int = 5
    max_drivers: int = 5
    tau_quantile: float = 0.95
    n_random_units: int = 20  # random non-candidate units per validation stay


@dataclass(frozen=True)
class Driver:
    """One selected driver: a unit or a merged run of blocks of one variable."""

    units: tuple[int, ...]
    modality: str
    key: str
    delta: float
    start_hour: int | None = None
    end_hour: int | None = None

    @property
    def direction(self) -> str:
        return "longer" if self.delta > 0 else "shorter"


def modality_slices(space: UnitSpace) -> dict[str, np.ndarray]:
    """Unit indices of each modality (the whole image counts with the image concepts)."""
    image = list(range(space.concept_slice.start, space.concept_slice.stop))
    if space.has_whole_image:
        image.append(space.whole_image_index)
    return {
        "timeseries": np.arange(space.n_ts),
        "static": np.arange(space.static_slice.start, space.static_slice.stop),
        "image": np.array(image, dtype=int),
    }


def candidate_units(scores: np.ndarray, space: UnitSpace, per_modality: int = 5) -> np.ndarray:
    """Step 1: the ``per_modality`` highest-|score| units of each modality (stable on ties)."""
    chosen: list[int] = []
    for units in modality_slices(space).values():
        order = np.argsort(-np.abs(scores[units]), kind="stable")[:per_modality]
        chosen.extend(int(units[i]) for i in order)
    return np.array(sorted(chosen), dtype=int)


def fit_tau(
    scorers: list[StayScorer],
    candidates: list[np.ndarray],
    cfg: SelectionConfig,
    seed: int = 666,
) -> float:
    """``tau``: the 0.95 quantile of |Δ| over random non-candidate units of validation stays."""
    rng = np.random.default_rng(seed)
    pooled: list[np.ndarray] = []
    for scorer, cand in zip(scorers, candidates, strict=True):
        pool = np.setdiff1d(np.arange(len(scorer.space)), cand)
        if len(pool) == 0:
            continue
        units = rng.choice(pool, size=min(cfg.n_random_units, len(pool)), replace=False)
        pooled.append(np.abs(scorer.removal_delta(units)))
    if not pooled:
        raise ValueError("no validation stay gave a non-candidate unit to fit tau on")
    return float(np.quantile(np.concatenate(pooled), cfg.tau_quantile))


def _merge_runs(units: list[int], deltas: dict[int, float], space: UnitSpace) -> list[list[int]]:
    """Group kept time-series blocks of one variable that touch and push the same way."""
    n_blocks = len(space.blocks)
    kept = set(units)
    groups: list[list[int]] = []
    seen: set[int] = set()
    for u in sorted(kept):
        if u in seen:
            continue
        run = [u]
        seen.add(u)
        if u < space.n_ts:
            nxt = u + 1
            while (
                nxt in kept
                and nxt // n_blocks == u // n_blocks  # the same variable
                and np.sign(deltas[nxt]) == np.sign(deltas[run[-1]])
            ):
                run.append(nxt)
                seen.add(nxt)
                nxt += 1
        groups.append(run)
    return groups


def select_drivers(
    scorer: StayScorer,
    scores: np.ndarray,
    tau: float,
    cfg: SelectionConfig = SelectionConfig(),  # noqa: B008 - frozen dataclass
) -> list[Driver]:
    """Steps 1 and 2 for one stay; at most ``cfg.max_drivers``, largest |Δ| first."""
    space = scorer.space
    candidates = candidate_units(scores, space, cfg.per_modality)
    delta = dict(zip(candidates.tolist(), scorer.removal_delta(candidates).tolist(), strict=True))
    kept = [u for u in candidates.tolist() if abs(delta[u]) >= tau and delta[u] != 0.0]
    drivers: list[Driver] = []
    for run in _merge_runs(kept, delta, space):
        first = space.units[run[0]]
        if len(run) == 1:
            value = delta[run[0]]
            start, end = first.start_hour, first.end_hour
        else:
            removed = np.zeros((1, len(space)), dtype=bool)
            removed[0, run] = True
            value = float(scorer.base_logit() - scorer.logits(removed)[0])
            start, end = first.start_hour, space.units[run[-1]].end_hour
            if abs(value) < tau:
                continue
        drivers.append(Driver(tuple(run), first.modality, first.key, float(value), start, end))
    drivers.sort(key=lambda d: (-abs(d.delta), d.units))
    return drivers[: cfg.max_drivers]
