"""Aggregate-only reporting for scripts that touch real data (CLAUDE.md A3.2).

Scripts that run on credentialed data print counts, shapes, metrics and summary
statistics only, never identifiers or rows, and suppress counts below 10. Every such
printout goes through these helpers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

MIN_CELL = 10


def suppress(n: int, min_cell: int = MIN_CELL) -> int | str:
    """Return ``n`` if it is at least ``min_cell``, else the string ``"<min_cell"``."""
    n = int(n)
    return n if n >= min_cell else f"<{min_cell}"


def suppress_counts(counts: Mapping[str, int], min_cell: int = MIN_CELL) -> dict[str, int | str]:
    """Apply :func:`suppress` to every value of a mapping."""
    return {key: suppress(value, min_cell) for key, value in counts.items()}


def safe_share(numerator: int, denominator: int, min_cell: int = MIN_CELL) -> float | str:
    """A proportion, suppressed when either cell (``numerator`` or the rest) is small.

    A share computed from a small numerator would reveal a small count, so both the
    numerator and ``denominator - numerator`` must reach ``min_cell``.
    """
    numerator, denominator = int(numerator), int(denominator)
    if denominator <= 0:
        return "n/a"
    if numerator < min_cell or (denominator - numerator) < min_cell:
        return f"suppressed (a cell < {min_cell})"
    return round(numerator / denominator, 4)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):  # numpy scalars
        return _jsonable(value.item())
    return value


def format_aggregates(payload: Mapping[str, Any]) -> str:
    """Pretty JSON for console output of an aggregate payload."""
    return json.dumps(_jsonable(payload), indent=2, sort_keys=False, default=str)
