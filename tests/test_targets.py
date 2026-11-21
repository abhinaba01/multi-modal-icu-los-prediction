"""Long-stay threshold: train-split Q3 until frozen; full runs require the frozen value."""

from __future__ import annotations

import pandas as pd
import pytest

from fate_icu.data.contract import DEFAULT_CONTRACT_PATH, Contract
from fate_icu.data.targets import (
    ThresholdNotFrozenError,
    long_stay_labels,
    resolve_long_stay_threshold,
    round_los,
    train_q3,
)
from fate_icu.tabular.run import load_run_config
from fate_icu.utils.env import REPO_ROOT


@pytest.fixture()
def los_split():
    los = pd.Series([1.111, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0], index=range(7))
    split = pd.Series(["train"] * 5 + ["test", "val"], index=range(7))
    return los, split


def test_q3_uses_train_stays_only(los_split, contract):
    los, split = los_split
    assert train_q3(los, split, contract) == pytest.approx(
        pd.Series([1.11, 2.0, 3.0, 4.0, 5.0]).quantile(0.75)
    )


def test_rounding_and_strict_inequality(contract):
    los = pd.Series([4.004, 4.006, 4.0])
    assert round_los(los, contract).tolist() == [4.0, 4.01, 4.0]
    assert long_stay_labels(los, 4.0, contract).tolist() == [0, 1, 0]


def _unfrozen_contract() -> Contract:
    import yaml

    raw = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["target"]["long_stay"].update(threshold_days=None, frozen_from=None)
    return Contract.model_validate(raw)


def test_unfrozen_threshold_is_computed_or_refused(los_split):
    contract = _unfrozen_contract()  # the repository's contract is frozen (4.1 d); build one that is not
    los, split = los_split
    value, source = resolve_long_stay_threshold(los, split, contract, require_frozen=False)
    assert source == "train_q3" and value == train_q3(los, split, contract)
    with pytest.raises(ThresholdNotFrozenError):
        resolve_long_stay_threshold(los, split, contract, require_frozen=True)


def test_frozen_threshold_wins(los_split):
    import yaml

    raw = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    raw["target"]["long_stay"].update(threshold_days=4.11, frozen_from="tabular_full_x")
    frozen = Contract.model_validate(raw)
    los, split = los_split
    assert resolve_long_stay_threshold(los, split, frozen, require_frozen=True) == (4.11, "frozen")


def test_full_config_requires_frozen_threshold():
    assert load_run_config(REPO_ROOT / "configs/tabular/full.yaml").require_frozen_threshold
    assert not load_run_config(REPO_ROOT / "configs/tabular/smoke.yaml").require_frozen_threshold
