"""Small helpers for writing result tables to disk as CSV and LaTeX."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it.

    Args:
        path: Directory path.

    Returns:
        The path as a :class:`Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_table(df: pd.DataFrame, out_dir: str | Path, name: str, float_format: str = "%.3f") -> None:
    """Write a DataFrame to ``<out_dir>/<name>.csv`` and ``.tex``.

    Args:
        df: Table to write (index preserved).
        out_dir: Output directory (created if needed).
        name: Base file name (no extension).
        float_format: Float format for both outputs.
    """
    directory = ensure_dir(out_dir)
    csv_path = directory / f"{name}.csv"
    df.to_csv(csv_path, float_format=float_format)
    try:
        (directory / f"{name}.tex").write_text(df.to_latex(float_format=float_format), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - latex is a nicety, never fatal
        LOGGER.warning("Could not write LaTeX for %s: %s", name, exc)
    LOGGER.info("wrote %s", csv_path)
