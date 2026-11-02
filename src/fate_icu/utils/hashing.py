"""Content hashes that are stable across machines and line-ending conventions.

Git on Windows may check text files out with CRLF line endings, so every hash here
normalises CRLF to LF before hashing. The same snapshot then hashes identically on the
laptop and the lab machine.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Directory names never part of a snapshot (caches written by running the code).
CACHE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".ruff_cache"})
CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _normalised_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def file_sha256(path: str | Path) -> str:
    """SHA-256 of one file with CRLF normalised to LF."""
    return hashlib.sha256(_normalised_bytes(Path(path))).hexdigest()


def tree_sha256(root: str | Path, exclude_top_level: frozenset[str] = frozenset()) -> str:
    """SHA-256 over every file under ``root`` (relative path + normalised content).

    Args:
        root: Directory to hash.
        exclude_top_level: Names of top-level entries to skip (e.g. run outputs).

    Returns:
        Hex digest. Cache folders and compiled Python files are always skipped.
    """
    root = Path(root)
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in exclude_top_level:
            continue
        if any(part in CACHE_DIR_NAMES for part in rel.parts) or path.suffix in CACHE_SUFFIXES:
            continue
        files.append(rel)
    for rel in sorted(files, key=lambda p: p.as_posix()):
        digest.update(rel.as_posix().encode("utf-8") + b"\0")
        digest.update(_normalised_bytes(root / rel) + b"\0")
    return digest.hexdigest()
