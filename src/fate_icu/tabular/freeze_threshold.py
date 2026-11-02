"""Print the train-split Q3 of LOS so it can be frozen into ``configs/contract.yaml``.

    python -m fate_icu.tabular.freeze_threshold --config configs/tabular/full.yaml

Run once in the lab; paste the output back. The value goes into
``target.long_stay.threshold_days`` (with the run id in ``frozen_from``) by a reviewed
commit, never automatically.
"""

from __future__ import annotations

import argparse

from fate_icu.data.contract import load_contract
from fate_icu.data.targets import long_stay_labels, train_q3
from fate_icu.tabular.api import load_splits, load_tables
from fate_icu.tabular.run import load_run_config
from fate_icu.utils.aggregates import format_aggregates, safe_share, suppress
from fate_icu.utils.manifest import new_run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    contract = load_contract()
    tables = load_tables(config.source, contract=contract, cache=config.cache, strict_sofa=config.strict_sofa)
    split = load_splits(tables.cohort, contract)
    q3 = train_q3(tables.los, split, contract)
    train = split == "train"
    y = long_stay_labels(tables.los, q3, contract)
    print(
        format_aggregates(
            {
                "run_id": new_run_id("freeze_threshold"),
                "train_q3_days": round(q3, 4),
                "train_stays": suppress(int(train.sum())),
                "train_long_stay_share": safe_share(int(y[train].sum()), int(train.sum())),
                "all_stays_long_share_at_this_threshold": safe_share(int(y.sum()), len(y)),
                "contract_sha256": contract.sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
