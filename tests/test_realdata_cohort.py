"""Lab only: the full-data cohort reproduces the internship's recorded counts.

The internship's metrics are lost; legacy_los/README.md recorded the cohort flow,
whole-cohort Q3 and long-stay prevalence on MIMIC-IV v3.1. These are the reproduction
targets in ``contract.reproduction_targets``. Prints aggregates only.
"""

from __future__ import annotations

import pytest

from fate_icu.tabular.api import load_tables


@pytest.mark.realdata
def test_cohort_flow_reproduces_the_internship(contract, settings):
    targets = contract.reproduction_targets
    tables = load_tables("postgres", contract=contract, settings=settings, cache=True)
    for step, expected in targets.cohort_flow.items():
        assert abs(tables.flow[step] - expected) <= targets.tolerance.counts, (
            step,
            tables.flow[step],
            expected,
        )


@pytest.mark.realdata
def test_whole_cohort_q3_and_prevalence_match(contract, settings):
    targets = contract.reproduction_targets
    tables = load_tables("postgres", contract=contract, settings=settings, cache=True)
    los = tables.los.round(contract.target.regression_rounding_dp)
    q3 = float(los.quantile(contract.target.long_stay.quantile))  # the internship's whole-cohort rule
    prevalence = float((los > q3).mean())
    assert abs(q3 - targets.whole_cohort_q3_days) <= targets.tolerance.q3_days, q3
    assert abs(prevalence - targets.long_prevalence) <= targets.tolerance.long_prevalence, prevalence
