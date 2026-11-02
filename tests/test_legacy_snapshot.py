"""legacy_los/ stays exactly the snapshot recorded in the contract (CLAUDE.md A3.6)."""

from __future__ import annotations

from pathlib import Path

from fate_icu.tabular._legacy import LEGACY_ROOT, ensure_legacy_importable, legacy_tree_sha256


def test_legacy_snapshot_is_unchanged(contract):
    assert legacy_tree_sha256() == contract.provenance.legacy_tree_sha256, (
        "legacy_los/ differs from the snapshot in configs/contract.yaml; it must never be edited"
    )


def test_no_claude_md_under_legacy():
    assert not [p for p in LEGACY_ROOT.rglob("*") if p.name.upper() == "CLAUDE.MD"]


def test_src_resolves_to_the_legacy_package():
    ensure_legacy_importable()
    import src

    assert LEGACY_ROOT.resolve() in Path(src.__file__).resolve().parents
