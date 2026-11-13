"""Machine-specific settings read from ``.env`` and the environment.

Paths, the database URL and the device are never hard-coded: each machine has its own
git-ignored ``.env`` (template: ``.env.example``). Relative paths resolve against the
repository root.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

REPO_ROOT = Path(__file__).resolve().parents[3]

Machine = Literal["laptop", "lab", "gcp", "kaggle"]

# Windows sets these to the roots of OneDrive-synced folders.
ONEDRIVE_VARIABLES = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")


class Settings(BaseModel):
    """Resolved settings for this machine."""

    machine: Machine = "laptop"
    db_url: str | None = None
    data_dir: Path
    artifacts_dir: Path
    outputs_dir: Path
    reports_dir: Path
    checkpoints_dir: Path
    device: Literal["auto", "cpu", "cuda"] = "auto"
    mimic_demo_dir: Path | None = None
    mimic_raw_dir: Path | None = None
    mimic_cxr_dir: Path | None = None
    icd9_gem_path: Path | None = None
    rag_dir: Path
    # NCBI E-utilities contact and key (Stage 5): secrets, never printed or logged.
    ncbi_email: SecretStr | None = None
    ncbi_api_key: SecretStr | None = None

    @property
    def db_url_redacted(self) -> str | None:
        """The database URL without user name or password (safe to log)."""
        return redact_url(self.db_url) if self.db_url else None


def _resolve(value: str | None, default: str | None = None) -> Path | None:
    raw = value if value not in (None, "") else default
    if raw in (None, ""):
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Read ``.env`` (without overriding variables already set) and build :class:`Settings`.

    Args:
        env_file: Optional path to a dotenv file; defaults to ``<repo>/.env``.

    Returns:
        The resolved settings.
    """
    load_dotenv(Path(env_file) if env_file else REPO_ROOT / ".env", override=False)
    env = os.environ
    artifacts_dir = _resolve(env.get("FATE_ARTIFACTS_DIR"), "artifacts")
    return Settings(
        machine=env.get("FATE_MACHINE") or "laptop",
        db_url=env.get("FATE_DB_URL") or None,
        data_dir=_resolve(env.get("FATE_DATA_DIR"), "data"),
        artifacts_dir=artifacts_dir,
        outputs_dir=_resolve(env.get("FATE_OUTPUTS_DIR"), "outputs"),
        reports_dir=_resolve(env.get("FATE_REPORTS_DIR"), "reports"),
        checkpoints_dir=_resolve(env.get("FATE_CHECKPOINTS_DIR"), "checkpoints"),
        device=env.get("FATE_DEVICE") or "auto",
        mimic_demo_dir=_resolve(env.get("MIMIC_DEMO_DIR")),
        mimic_raw_dir=_resolve(env.get("MIMIC_RAW_DIR")),
        mimic_cxr_dir=_resolve(env.get("MIMIC_CXR_DIR")),
        icd9_gem_path=_resolve(env.get("ICD9_GEM_PATH")),
        rag_dir=_resolve(env.get("FATE_RAG_DIR")) or artifacts_dir / "rag",
        ncbi_email=env.get("NCBI_EMAIL") or None,
        ncbi_api_key=env.get("NCBI_API_KEY") or None,
    )


def under_onedrive(path: str | Path, environ: Mapping[str, str] | None = None) -> bool:
    """True when ``path`` lies inside a OneDrive-synced folder (Windows sets ``%OneDrive%``).

    Args:
        path: The folder to check.
        environ: Environment to read the OneDrive roots from (defaults to ``os.environ``).
    """
    environ = os.environ if environ is None else environ
    target = Path(path).resolve()
    for name in ONEDRIVE_VARIABLES:
        root = environ.get(name)
        if not root:
            continue
        try:
            target.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return True
    return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached :func:`load_settings` for application code (tests call ``load_settings``)."""
    return load_settings()


def redact_url(url: str) -> str:
    """Drop the user name and password from a URL, keeping scheme, host, port and path.

    Args:
        url: e.g. ``postgresql+psycopg://user:secret@localhost:5432/mimiciv_demo``.

    Returns:
        e.g. ``postgresql+psycopg://localhost:5432/mimiciv_demo``.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
