"""Stage 1 on the local open demo database (``demo`` marker, laptop only).

No open X-rays link to the demo patients, so the metadata is synthetic, keyed to the
demo's real ICU stays.
"""

from __future__ import annotations

import pytest

from fate_icu.data.cxr import link as link_cli
from fate_icu.data.cxr.linkage import link
from fate_icu.data.cxr.report import check_planted_answers
from fate_icu.data.cxr.synthetic import SCENARIOS, generate_cxr_metadata
from fate_icu.tabular.api import load_tables
from fate_icu.utils.env import REPO_ROOT

pytestmark = pytest.mark.demo


def test_planted_answers_on_the_demo_stays(settings, contract):
    cohort = load_tables("postgres", contract=contract, settings=settings, cache=True).cohort
    hours = contract.window.hours
    fake = generate_cxr_metadata(cohort, hours, 24.0, seed=0)
    result = check_planted_answers(link(fake.metadata, cohort, hours, 24.0).paired, fake.expected)
    assert result["all_correct"], result["wrong_by_scenario"]
    assert result["scenarios_planted"] == len(SCENARIOS)


def test_link_command_on_the_demo(settings, contract):
    report = link_cli.run(
        link_cli.load_config(REPO_ROOT / "configs" / "cxr" / "smoke.yaml"), contract, settings=settings
    )
    assert report["planted_answers"]["all_correct"]
    assert "linked_share_by_anchor_year_group" in report
