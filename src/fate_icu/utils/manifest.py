"""``run_manifest.json``: what produced a run (CLAUDE.md A4, reproducibility).

Every run records the git commit, its resolved config, seeds, package versions, the
contract and legacy-snapshot hashes, the data source and model revisions. Database URLs
are written without credentials.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from fate_icu.utils.env import REPO_ROOT, redact_url

MANIFEST_NAME = "run_manifest.json"

# Packages whose versions are recorded when installed (absent ones are skipped).
TRACKED_PACKAGES = (
    "fate-icu",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "pyarrow",
    "pydantic",
    "sqlalchemy",
    "psycopg",
    "torch",
    "torchxrayvision",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "captum",
    "faiss-cpu",
    "bm25s",
    "pystemmer",
    "tokenizers",
    "huggingface-hub",
    "pypdf",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """Current commit, branch and whether the working tree has uncommitted changes."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the tracked packages."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _scrub(value: Any) -> Any:
    """Recursively redact credentials from any URL-looking string."""
    if isinstance(value, Mapping):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and "://" in value and "@" in value:
        return redact_url(value)
    return value


def build_manifest(
    *,
    run_id: str,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    contract_sha256: str | None = None,
    legacy_tree_sha256: str | None = None,
    data: Mapping[str, Any] | None = None,
    model_revisions: Mapping[str, str] | None = None,
    device: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dictionary (see module docstring)."""
    return _scrub(
        {
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "packages": package_versions(),
            "config": dict(config),
            "seeds": list(seeds),
            "contract_sha256": contract_sha256,
            "legacy_tree_sha256": legacy_tree_sha256,
            "data": dict(data or {}),
            "model_revisions": dict(model_revisions or {}),
            "extra": dict(extra or {}),
        }
    )


def write_run_manifest(out_dir: str | Path, **kwargs: Any) -> Path:
    """Build the manifest and write it to ``<out_dir>/run_manifest.json``.

    Args:
        out_dir: Run output directory (created if needed).
        **kwargs: Passed to :func:`build_manifest`.

    Returns:
        Path of the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(**kwargs), indent=2, default=str), encoding="utf-8")
    return path


def new_run_id(name: str) -> str:
    """``<name>_<UTC timestamp>``, e.g. ``tabular_smoke_20261009T120000Z``."""
    return f"{name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
