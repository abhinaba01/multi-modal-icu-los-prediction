"""Targets from the contract: LOS in days and the long-stay label.

The long-stay threshold is the Q3 of ``round(los, 2)`` over the FaTE train split
(contract ``target.long_stay``). Once the lab value is frozen in ``contract.yaml``,
every stage uses that number; until then smoke runs compute it from their own train
split, and full runs refuse to start.
"""

from __future__ import annotations

import pandas as pd

from fate_icu.data.contract import Contract


class ThresholdNotFrozenError(RuntimeError):
    """A full run needs the frozen long-stay threshold from ``contract.yaml``."""


def round_los(los: pd.Series, contract: Contract) -> pd.Series:
    """Regression target: LOS in days, rounded as in the internship."""
    return los.astype(float).round(contract.target.regression_rounding_dp)


def train_q3(los: pd.Series, split: pd.Series, contract: Contract) -> float:
    """Q3 (contract quantile) of the rounded LOS over train stays only."""
    train = split.reindex(los.index) == "train"
    if not train.any():
        raise ValueError("no train stays to compute the long-stay threshold from")
    return float(round_los(los[train], contract).quantile(contract.target.long_stay.quantile))


def resolve_long_stay_threshold(
    los: pd.Series, split: pd.Series, contract: Contract, require_frozen: bool
) -> tuple[float, str]:
    """The threshold in days and where it came from (``frozen`` or ``train_q3``).

    Raises:
        ThresholdNotFrozenError: ``require_frozen`` and the contract value is still null.
    """
    spec = contract.target.long_stay
    if spec.threshold_days is not None:
        return float(spec.threshold_days), "frozen"
    if require_frozen:
        raise ThresholdNotFrozenError(
            "target.long_stay.threshold_days is null in configs/contract.yaml. Run "
            "`python -m fate_icu.tabular.freeze_threshold --config configs/tabular/full.yaml` "
            "in the lab and freeze the printed value first."
        )
    return train_q3(los, split, contract), "train_q3"


def long_stay_labels(los: pd.Series, threshold_days: float, contract: Contract) -> pd.Series:
    """1 = long stay (rounded LOS strictly above the threshold), 0 = short."""
    return (round_los(los, contract) > threshold_days).astype(int).rename("long_stay")
