"""Does an attribution method rank the units that really move the model? (Stage 4)

All functions take unit scores in the order of ``UnitSpace`` and a ``score_removed`` function
that returns the model's long-stay logit for a removal mask (``StayScorer.logits``), so they
measure the model through the same removal it was trained with.

* precision / recall at k against planted units (synthetic benchmark);
* deletion and insertion curves with AOPC, comprehensiveness and sufficiency at k, as
  DeYoung et al. (2020) define them, on the predicted probability of the long-stay class;
* the model-randomisation check of Adebayo et al. (2018): an attribution that does not change
  when the weights are randomised explains the data or the method, not the model.

Probabilities are of the stay's *predicted* class for the curves: a stay predicted long is
explained by how much deleting evidence lowers P(long), a stay predicted short by how much it
raises it, so "important" always means "the prediction falls apart without it".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

K_VALUES = (1, 3, 5)
ScoreRemoved = Callable[[np.ndarray], np.ndarray]  # bool [B, n_units] -> logits [B]


def precision_recall_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall of the ``k`` highest-|score| units against planted ``truth`` (bool)."""
    truth = np.asarray(truth, dtype=bool)
    if k < 1:
        raise ValueError("k must be at least 1")
    if not truth.any():
        raise ValueError("a stay without a planted unit has no recall")
    top = np.argsort(-np.abs(np.asarray(scores)), kind="stable")[:k]
    hits = int(truth[top].sum())
    return hits / k, hits / int(truth.sum())


def _predicted_probability(logit: np.ndarray, predicted_long: bool) -> np.ndarray:
    p_long = 1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=float)))
    return p_long if predicted_long else 1.0 - p_long


def _mask_of(order: np.ndarray, count: int, n_units: int) -> np.ndarray:
    mask = np.zeros(n_units, dtype=bool)
    mask[order[:count]] = True
    return mask


def ranking(scores: np.ndarray, usable: np.ndarray | None = None) -> np.ndarray:
    """Unit indices from most to least important by |score| (stable); ``usable`` limits the pool."""
    order = np.argsort(-np.abs(np.asarray(scores)), kind="stable")
    if usable is not None:
        order = order[np.asarray(usable, dtype=bool)[order]]
    return order


def curves(
    order: np.ndarray, n_units: int, score_removed: ScoreRemoved, max_steps: int | None = None
) -> dict[str, np.ndarray | float]:
    """Deletion and insertion curves along ``order`` and their areas.

    Deletion: remove the units one by one in order; the predicted-class probability should
    fall fast (a low area is good). Insertion: start from nothing and add units in order; it
    should rise fast (a high area is good).
    """
    steps = len(order) if max_steps is None else min(max_steps, len(order))
    removal = np.stack([_mask_of(order, j, n_units) for j in range(steps + 1)])
    full = score_removed(np.zeros((1, n_units), dtype=bool))[0]
    predicted_long = full > 0
    deleted = score_removed(removal)
    inserted = score_removed(~removal)
    p_full = float(_predicted_probability(full, predicted_long))
    deletion = _predicted_probability(deleted, predicted_long)
    insertion = _predicted_probability(inserted, predicted_long)
    x = np.arange(steps + 1) / max(steps, 1)
    return {
        "p_full": p_full,
        "predicted_long": bool(predicted_long),
        "deletion": deletion,
        "insertion": insertion,
        "aopc": float(np.mean(p_full - deletion[1:])) if steps else 0.0,  # mean drop over k = 1..steps
        "deletion_auc": float(np.trapezoid(deletion, x)),
        "insertion_auc": float(np.trapezoid(insertion, x)),
    }


def comprehensiveness_sufficiency(
    order: np.ndarray, n_units: int, score_removed: ScoreRemoved, ks: Sequence[int] = K_VALUES
) -> dict[int, dict[str, float]]:
    """DeYoung et al.: comprehensiveness = p(x) - p(x without the top k) (high is good);
    sufficiency = p(x) - p(only the top k kept) (low is good)."""
    full = score_removed(np.zeros((1, n_units), dtype=bool))[0]
    predicted_long = full > 0
    p_full = float(_predicted_probability(full, predicted_long))
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        k_eff = min(k, len(order))
        top = _mask_of(order, k_eff, n_units)
        pair = score_removed(np.stack([top, ~top]))
        without_top, only_top = _predicted_probability(pair, predicted_long)
        out[k] = {"comprehensiveness": p_full - float(without_top), "sufficiency": p_full - float(only_top)}
    return out


def random_order(n_units: int, usable: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    pool = np.flatnonzero(np.ones(n_units, dtype=bool) if usable is None else usable)
    return rng.permutation(pool)


def randomisation_agreement(trained_scores: np.ndarray, random_scores: np.ndarray) -> float:
    """Spearman correlation of |scores| from the trained and a weight-randomised model.

    Near 0 is what a method that explains the model should give; near 1 means the explanation
    does not depend on what the model learned (Adebayo et al., 2018).
    """
    a, b = np.abs(np.asarray(trained_scores)), np.abs(np.asarray(random_scores))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
