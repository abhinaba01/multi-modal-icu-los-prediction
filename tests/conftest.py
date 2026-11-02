"""Shared fixtures and marker handling.

* ``gpu``: skipped without a CUDA device.
* ``realdata``: credentialed data; runs only when ``FATE_MACHINE=lab``.
* ``demo``: the local ``mimiciv_demo`` database; laptop only, skipped until
  ``FATE_DB_URL`` is set. Once it is set, an unreachable database fails loudly.
"""

from __future__ import annotations

import pytest

from fate_icu.data.contract import load_contract
from fate_icu.data.splits import fate_splits
from fate_icu.data.synthetic import generate_synthetic_mimic
from fate_icu.data.timeseries.build import build
from fate_icu.data.timeseries.sources import FrameEventSource
from fate_icu.data.timeseries.variables import load_variables
from fate_icu.tabular.api import load_tables
from fate_icu.tabular.loaders import FrameLoader
from fate_icu.utils.device import cuda_available
from fate_icu.utils.env import load_settings


def pytest_collection_modifyitems(config, items):
    settings = load_settings()
    has_cuda = None
    for item in items:
        if "gpu" in item.keywords:
            has_cuda = cuda_available() if has_cuda is None else has_cuda
            if not has_cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "realdata" in item.keywords and settings.machine != "lab":
            item.add_marker(pytest.mark.skip(reason="credentialed data: runs only with FATE_MACHINE=lab"))
        if "demo" in item.keywords:
            if settings.machine != "laptop":
                item.add_marker(pytest.mark.skip(reason="demo tests run on the laptop only"))
            elif not settings.db_url:
                item.add_marker(
                    pytest.mark.skip(reason="FATE_DB_URL not set: load the demo first (docs/laptop_setup.md)")
                )


@pytest.fixture(scope="session")
def contract():
    return load_contract()


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def synth():
    return generate_synthetic_mimic(n_patients=60, seed=0)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, synth):
    """The synthetic tables written in the MIMIC-IV CSV layout, plus a mini GEM file."""
    return synth.write_csv_layout(tmp_path_factory.mktemp("synthetic_mimic"))


@pytest.fixture(scope="session")
def synth_tables(synth, synth_dir, contract):
    """Contract cohort, features and provenance built from the synthetic tables."""
    return load_tables(
        loader=FrameLoader(synth.tables), contract=contract, gem_path=synth.gem_path(synth_dir), cache=False
    )


@pytest.fixture(scope="session")
def variables():
    return load_variables()


@pytest.fixture(scope="session")
def synth_split(synth_tables, contract):
    return fate_splits(synth_tables.cohort.sort_index(), contract)


@pytest.fixture(scope="session")
def ts_build(synth, synth_tables, synth_split, contract, variables):
    """Stage 2 tensors built from the synthetic tables (every channel must carry signal)."""
    source = FrameEventSource(synth.tables, contract.window.hours)
    return build(
        source, synth_tables.cohort, synth_split, variables, contract.window.hours, require_signal=True
    )
