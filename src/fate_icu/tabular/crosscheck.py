"""Lab check: features from PostgreSQL equal those from the legacy CSV reader.

    python -m fate_icu.tabular.crosscheck --config configs/tabular/full.yaml

Builds the contract tables twice, from ``FATE_DB_URL`` and from the raw CSVs in
``MIMIC_RAW_DIR`` (what the internship read), and prints aggregates only: cohort sizes,
whether the stay sets agree, and the largest absolute difference per feature group.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fate_icu.data.contract import load_contract
from fate_icu.tabular.api import load_tables
from fate_icu.tabular.run import load_run_config
from fate_icu.utils.aggregates import format_aggregates, suppress


def compare(a: pd.DataFrame, b: pd.DataFrame, groups: dict[str, str]) -> dict[str, float | str]:
    """Largest absolute difference per feature group over the shared stays."""
    shared = a.index.intersection(b.index)
    out: dict[str, float | str] = {}
    for group in dict.fromkeys(groups.values()):
        cols = [c for c, g in groups.items() if g == group]
        left, right = a.loc[shared, cols], b.loc[shared, cols]
        numeric = left.select_dtypes("number").columns
        diff = (left[numeric] - right[numeric]).abs().to_numpy()
        worst = float(np.nanmax(diff)) if diff.size else 0.0
        mismatched = int(
            (left.drop(columns=numeric).astype(str) != right.drop(columns=numeric).astype(str)).sum().sum()
        )
        out[group] = worst if mismatched == 0 else f"{mismatched} categorical mismatches"
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    contract = load_contract()
    from_pg = load_tables("postgres", contract=contract, cache=config.cache, strict_sofa=config.strict_sofa)
    from_csv = load_tables("csv_raw", contract=contract, cache=config.cache, strict_sofa=config.strict_sofa)
    groups = contract.group_of(list(from_pg.features.columns))
    print(
        format_aggregates(
            {
                "stays_postgres": suppress(len(from_pg.cohort)),
                "stays_csv": suppress(len(from_csv.cohort)),
                "same_stay_set": bool(
                    from_pg.cohort.index.sort_values().equals(from_csv.cohort.index.sort_values())
                ),
                "flow_equal": from_pg.flow == from_csv.flow,
                "max_abs_difference_by_group": compare(from_pg.features, from_csv.features, groups),
            }
        )
    )


if __name__ == "__main__":
    main()
