"""Utilities: settings, hashing, seeding, device, manifests and aggregate suppression."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from fate_icu.utils.aggregates import format_aggregates, safe_share, suppress, suppress_counts
from fate_icu.utils.device import get_device
from fate_icu.utils.env import REPO_ROOT, load_settings, redact_url
from fate_icu.utils.hashing import file_sha256, tree_sha256
from fate_icu.utils.manifest import MANIFEST_NAME, build_manifest, write_run_manifest
from fate_icu.utils.seed import run_seeds, seed_everything


def test_redact_url():
    assert (
        redact_url("postgresql+psycopg://u:p@db.local:5432/mimiciv")
        == "postgresql+psycopg://db.local:5432/mimiciv"
    )
    assert redact_url("postgresql://localhost/mimiciv_demo") == "postgresql://localhost/mimiciv_demo"


def test_settings_resolve_relative_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", "outputs_x")
    monkeypatch.setenv("FATE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("FATE_MACHINE", "lab")
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.outputs_dir == REPO_ROOT / "outputs_x"
    assert settings.artifacts_dir == tmp_path
    assert settings.machine == "lab"


def test_hash_ignores_line_endings_and_caches(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, eol in ((a, b"\n"), (b, b"\r\n")):
        (root / "pkg" / "__pycache__").mkdir(parents=True)
        (root / "pkg" / "m.py").write_bytes(b"x = 1" + eol + b"y = 2" + eol)
    (b / "pkg" / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"\x00")
    (b / "results").mkdir()
    (b / "results" / "t.tex").write_text("output")
    assert tree_sha256(a) == tree_sha256(b, exclude_top_level=frozenset({"results"}))
    assert file_sha256(a / "pkg" / "m.py") == file_sha256(b / "pkg" / "m.py")
    (b / "pkg" / "m.py").write_bytes(b"x = 2\n")
    assert tree_sha256(a) != tree_sha256(b, exclude_top_level=frozenset({"results"}))


def test_seeding_is_reproducible():
    seed_everything(5)
    first = (random.random(), np.random.rand())
    seed_everything(5)
    assert (random.random(), np.random.rand()) == first
    assert run_seeds(666) == [666, 667, 668, 669, 670]
    with pytest.raises(ValueError):
        run_seeds(1, 0)


def test_device():
    assert get_device("cpu") == "cpu"
    assert get_device("auto") in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        get_device("tpu")


def test_small_counts_are_suppressed():
    assert suppress(9) == "<10" and suppress(10) == 10 and suppress(0) == "<10"
    assert suppress_counts({"a": 3, "b": 300}) == {"a": "<10", "b": 300}
    assert safe_share(5, 1000) == "suppressed (a cell < 10)"
    assert safe_share(995, 1000) == "suppressed (a cell < 10)"
    assert safe_share(250, 1000) == 0.25
    assert json.loads(format_aggregates({"x": np.float64(1.5), "y": float("nan")})) == {"x": 1.5, "y": None}


def test_manifest_records_versions_and_hides_credentials(tmp_path):
    path = write_run_manifest(
        tmp_path,
        run_id="r1",
        config={"db": "postgresql+psycopg://alice:s3cret@host:5432/db", "rounds": 3},
        seeds=[1, 2],
        contract_sha256="c" * 64,
        data={"url": "postgresql://bob:pw@host/db"},
    )
    assert path.name == MANIFEST_NAME
    text = path.read_text(encoding="utf-8")
    assert "s3cret" not in text and "alice" not in text and "pw@" not in text
    manifest = json.loads(text)
    assert manifest["seeds"] == [1, 2] and manifest["config"]["rounds"] == 3
    assert {"commit", "dirty", "branch"} <= set(manifest["git"])
    assert "numpy" in manifest["packages"]
    assert build_manifest(run_id="r", config={}, seeds=[0])["python"].startswith("3.")
