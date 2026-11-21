"""H4: does two-step driver selection beat gradient-only and random selection? (Stage 7)

For each explained stay the model is probed with the same removal as in training (``score_removed``).
Three selections of k units are compared on the *same* stay:

* ``two_step``: the first k units of the packet's driver list (gradient score, then deletion check, then tau);
* ``gradient``: the k units with the largest |attribution score| (no deletion check, no tau);
* ``random``: k random usable units.

Comprehensiveness at k (DeYoung et al., 2020) is the drop in the predicted-class probability when the
selected units are removed together; higher is better. Sufficiency is the drop when only they are kept; lower
is better. The registered test (``docs/hypotheses.md``, H4) compares comprehensiveness at k = 3 with two
paired bootstraps and takes the larger p-value (an intersection test, so no extra correction is needed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fate_icu.eval.attr_faithfulness import (
    ScoreRemoved,
    comprehensiveness_sufficiency,
    random_order,
    ranking,
)
from fate_icu.eval.stats import paired_mean_difference

METHODS = ("two_step", "gradient", "random")


@dataclass
class StaySelection:
    """What H4 needs from one explained stay."""

    n_units: int
    score_removed: ScoreRemoved  # bool [B, n_units] -> logits [B]
    two_step: np.ndarray  # unit indices of the packet's drivers, in driver order
    gradient_scores: np.ndarray  # [n_units] attribution scores before the deletion check
    usable: np.ndarray | None = None  # bool [n_units]: units that may be selected


def per_stay_values(
    stays: Sequence[StaySelection], *, k: int = 3, seed: int = 666
) -> dict[str, dict[str, np.ndarray]]:
    """``{"comprehensiveness": {method: [N]}, "sufficiency": {method: [N]}}`` at ``k``."""
    rng = np.random.default_rng(seed)
    out = {m: {meth: [] for meth in METHODS} for m in ("comprehensiveness", "sufficiency")}
    for s in stays:
        orders = {
            "two_step": np.asarray(s.two_step, dtype=int),
            "gradient": ranking(s.gradient_scores, s.usable),
            "random": random_order(s.n_units, s.usable, rng),
        }
        for method, order in orders.items():
            got = comprehensiveness_sufficiency(order, s.n_units, s.score_removed, ks=(k,))[k]
            out["comprehensiveness"][method].append(got["comprehensiveness"])
            out["sufficiency"][method].append(got["sufficiency"])
    return {m: {meth: np.asarray(v) for meth, v in d.items()} for m, d in out.items()}


def h4_test(
    comprehensiveness: dict[str, np.ndarray], patients: np.ndarray, *, n_boot: int = 1000, seed: int = 666
) -> dict[str, object]:
    """The registered H4 comparison on comprehensiveness: two-step minus gradient-only and minus random.

    ``p_value`` is the larger of the two bootstrap p-values; ``higher_than_both`` is true when both
    differences are positive. The decision (Holm across hypotheses) is taken by the caller.
    """
    versus = {
        other: paired_mean_difference(
            comprehensiveness["two_step"], comprehensiveness[other], patients, n_boot=n_boot, seed=seed
        )
        for other in ("gradient", "random")
    }
    return {
        "versus": versus,
        "p_value": max(v["p_value"] for v in versus.values()),
        "higher_than_both": all(v["difference"] > 0 for v in versus.values()),
    }
