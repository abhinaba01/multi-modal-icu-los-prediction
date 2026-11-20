"""SHA-256 manifests for moving big folders between machines.

    python -m fate_icu.utils.checksums make <folder> [--exclude GLOB ...]
    python -m fate_icu.utils.checksums check <folder>

``make`` writes ``<folder>/SHA256SUMS.txt``: one line per file, ``<sha256>  <path>``, where the
path is relative to the folder and uses ``/`` (the format of ``sha256sum``), sorted by path.
The manifest itself and anything matching an ``--exclude`` pattern are left out. A pattern is
matched against the whole relative path and against each folder or file name in it, so
``rag_dryrun_*`` skips those folders. Hashes are of the raw bytes (no line-ending changes).

``check`` hashes the files again and reports the ones that are missing, changed, or present but
not in the manifest. Copy the manifest together with the folder, run ``check`` on the other
machine, and the copy is proven intact before the original is wiped. It exits with an error
unless every listed file is present and unchanged; extra files are only reported.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.hashing import binary_sha256

MANIFEST_NAME = "SHA256SUMS.txt"
# "<sha256>  <path>" (sha256sum, two spaces) or "<sha256> <path>" (PhysioNet's SHA256SUMS.txt, one space);
# an optional "*" before the path marks binary mode.
_LINE = re.compile(r"^([0-9a-f]{64}) {1,2}\*?(.+)$")
REPORTED = 20  # file names listed per problem kind


def _excluded(relative: PurePosixPath, patterns: Iterable[str]) -> bool:
    text = relative.as_posix()
    return any(
        fnmatch.fnmatch(text, pattern) or any(fnmatch.fnmatch(part, pattern) for part in relative.parts)
        for pattern in patterns
    )


def list_files(
    folder: str | Path, *, manifest_name: str = MANIFEST_NAME, exclude: Sequence[str] = ()
) -> list[PurePosixPath]:
    """Every file under ``folder`` (relative, sorted), without the manifest and excluded paths."""
    folder = Path(folder)
    found = []
    for root, dirs, files in os.walk(folder):
        relative_root = PurePosixPath(Path(root).relative_to(folder).as_posix())
        dirs[:] = sorted(d for d in dirs if not _excluded(relative_root / d, exclude))
        for name in files:
            relative = relative_root / name
            if relative == PurePosixPath(manifest_name) or _excluded(relative, exclude):
                continue
            found.append(relative)
    return sorted(found)


def make_manifest(
    folder: str | Path,
    *,
    exclude: Sequence[str] = (),
    manifest_name: str = MANIFEST_NAME,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Hash every file of ``folder`` and write ``<folder>/<manifest_name>`` (atomically)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    files = list_files(folder, manifest_name=manifest_name, exclude=exclude)
    if not files:
        raise ValueError(f"{folder} has no files to list")
    lines, total = [], 0
    for number, relative in enumerate(files, start=1):
        lines.append(f"{binary_sha256(folder / relative)}  {relative.as_posix()}")
        total += (folder / relative).stat().st_size
        if log and (number % 100 == 0 or number == len(files)):
            log(f"hashed {number:,} of {len(files):,} files ({total / 1e9:,.2f} GB)")
    path = folder / manifest_name
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    return {"manifest": str(path), "files": len(files), "gigabytes": round(total / 1e9, 3)}


def read_manifest(path: str | Path) -> dict[str, str]:
    """``{relative path: sha256}`` from a manifest; a malformed or repeated line is an error."""
    entries: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None:
            raise ValueError(f"{path} line {number} is not '<sha256>  <path>'")
        digest, relative = match.groups()
        if relative in entries:
            raise ValueError(f"{path} lists {relative} twice")
        entries[relative] = digest
    return entries


def check_manifest(
    folder: str | Path,
    *,
    manifest_name: str = MANIFEST_NAME,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compare the files of ``folder`` with its manifest (see the module text)."""
    folder = Path(folder)
    manifest = folder / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} is missing: copy the manifest together with the folder")
    expected = read_manifest(manifest)
    missing, changed = [], []
    for number, (relative, digest) in enumerate(expected.items(), start=1):
        path = folder / relative
        if not path.is_file():
            missing.append(relative)
        elif binary_sha256(path) != digest:
            changed.append(relative)
        if log and (number % 100 == 0 or number == len(expected)):
            log(f"checked {number:,} of {len(expected):,} files")
    present = {p.as_posix() for p in list_files(folder, manifest_name=manifest_name)}
    extra = sorted(present - set(expected))
    return {
        "files_listed": len(expected),
        "files_ok": len(expected) - len(missing) - len(changed),
        "missing": missing[:REPORTED],
        "n_missing": len(missing),
        "changed": changed[:REPORTED],
        "n_changed": len(changed),
        "extra_files_not_in_manifest": len(extra),
        "extra_examples": extra[:5],
        "ok": not missing and not changed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="write SHA256SUMS.txt for a folder")
    make.add_argument("folder")
    make.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB", help="leave out matching paths"
    )
    check = commands.add_parser("check", help="verify a folder against its SHA256SUMS.txt")
    check.add_argument("folder")
    for sub in (make, check):
        sub.add_argument(
            "--manifest", default=MANIFEST_NAME, help=f"manifest file name (default {MANIFEST_NAME})"
        )
    args = parser.parse_args(argv)

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    if args.command == "make":
        result = make_manifest(args.folder, exclude=args.exclude, manifest_name=args.manifest, log=progress)
    else:
        result = check_manifest(args.folder, manifest_name=args.manifest, log=progress)
    print(format_aggregates(result))
    if args.command == "check" and not result["ok"]:
        raise SystemExit("the copy does not match its manifest: copy the listed files again")


if __name__ == "__main__":
    main()
