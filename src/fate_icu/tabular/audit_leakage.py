"""How often a value the legacy features use was stored only after t0 (aggregates only).

The internship features select events by ``charttime`` (``starttime`` for inputevents)
and never check ``storetime``. A lab drawn at hour 23 whose result posted at hour 25 is
therefore counted although it was not available at t0. This audit measures that share,
per source table, over the rows the legacy windows select for the final cohort:

    python -m fate_icu.tabular.audit_leakage --config configs/tabular/smoke.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from fate_icu.data.contract import Contract, load_contract
from fate_icu.tabular._legacy import legacy_config
from fate_icu.tabular.api import load_tables, make_loader
from fate_icu.tabular.loaders import legacy_itemid_filters
from fate_icu.tabular.run import load_run_config
from fate_icu.utils.aggregates import format_aggregates, safe_share, suppress
from fate_icu.utils.env import load_settings

# table -> (join key, event-time column)
AUDITED = {
    "chartevents": ("stay_id", "charttime"),
    "labevents": ("hadm_id", "charttime"),
    "outputevents": ("stay_id", "charttime"),
    "inputevents": ("stay_id", "starttime"),
}


def audit_storetime(loader: Any, cohort: pd.DataFrame, contract: Contract) -> dict[str, Any]:
    """Share of in-window rows with ``storetime`` after t0, per table.

    Args:
        loader: Legacy-interface loader.
        cohort: Final cohort (index ``stay_id``) with ``hadm_id`` and ``intime``.
        contract: The frozen contract (window W).

    Returns:
        Aggregates per table: rows in the window, rows stored after t0, their share,
        stays with at least one such row, and the median storage delay past t0.
    """
    hours = contract.window.hours
    filters = legacy_itemid_filters()
    keys = cohort.reset_index()[["stay_id", "hadm_id", "intime"]].copy()
    keys["intime"] = pd.to_datetime(keys["intime"])
    report: dict[str, Any] = {}
    for table, (join_key, time_col) in AUDITED.items():
        key_cols = ["stay_id", "intime"] if join_key == "stay_id" else ["hadm_id", "stay_id", "intime"]
        n_window = n_late = n_missing = 0
        late_stays: set[int] = set()
        delays: list[np.ndarray] = []
        usecols = list(dict.fromkeys([join_key, "itemid", time_col, "storetime"]))
        for chunk in loader.iter_chunks(table, usecols=usecols):
            chunk = chunk[chunk["itemid"].isin(filters[table])]
            if chunk.empty:
                continue
            chunk = chunk.assign(
                **{time_col: pd.to_datetime(chunk[time_col]), "storetime": pd.to_datetime(chunk["storetime"])}
            )
            merged = chunk.merge(keys[key_cols], on=join_key, how="inner")
            offset = (merged[time_col] - merged["intime"]).dt.total_seconds() / 3600.0
            # union of the legacy windows (intime, t0] and [intime, t0)
            in_window = merged[(offset >= 0) & (offset <= hours)]
            t0 = in_window["intime"] + pd.Timedelta(hours=hours)
            late = in_window["storetime"] > t0
            n_window += len(in_window)
            n_missing += int(in_window["storetime"].isna().sum())
            n_late += int(late.sum())
            late_stays.update(in_window.loc[late, "stay_id"].astype("int64").tolist())
            delays.append(
                ((in_window.loc[late, "storetime"] - t0[late]).dt.total_seconds() / 3600.0).to_numpy()
            )
        delay = np.concatenate(delays) if delays else np.array([])
        report[table] = {
            "rows_in_window": suppress(n_window),
            "rows_stored_after_t0": suppress(n_late),
            "share_stored_after_t0": safe_share(n_late, n_window),
            "rows_without_storetime": suppress(n_missing),
            "stays_with_a_late_row": suppress(len(late_stays)),
            "share_of_stays_with_a_late_row": safe_share(len(late_stays), len(cohort)),
            "median_hours_stored_after_t0": round(float(np.median(delay)), 2) if len(delay) >= 10 else "n<10",
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    contract = load_contract()
    settings = load_settings()
    tables = load_tables(
        config.source,
        contract=contract,
        settings=settings,
        cache=config.cache,
        strict_sofa=config.strict_sofa,
    )
    cfg = legacy_config(contract, gem_path=settings.icd9_gem_path)
    loader = make_loader(config.source, contract, settings, cfg)
    print(
        format_aggregates(
            {
                "stays": suppress(len(tables.cohort)),
                "storetime_audit": audit_storetime(loader, tables.cohort, contract),
            }
        )
    )


if __name__ == "__main__":
    main()
