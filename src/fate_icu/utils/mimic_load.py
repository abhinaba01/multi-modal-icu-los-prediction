"""Download MIMIC-IV, load it into PostgreSQL and build the derived concepts (Stage 0, 0.1).

    python -m fate_icu.utils.mimic_load --step all --user <PHYSIONET_USERNAME>

Runs on the GCP VM only (CLAUDE.md A2, A3): credentialed data is never downloaded to the laptop.
It automates the steps of docs/lab_runbook.md 0.1, one step at a time so each can be repeated:

* ``preflight``  tools, free disk space and the mimic-code checkout (cloned at the pinned tag);
* ``download``   ``wget`` from PhysioNet; wget asks for your password itself (never passed on a
                 command line or stored), and ``-c -N`` resumes a broken download;
* ``verify``     every file against PhysioNet's ``SHA256SUMS.txt``;
* ``load``       create the database and run mimic-code's create, load, constraint and index scripts;
* ``concepts``   mimic-code's derived concepts (``mimiciv_derived``, including ``sofa``);
* ``check``      row counts, printed as counts only (no rows, no identifiers).

``--dry-run`` prints every command without running it. A finished load is detected and skipped
unless ``--force`` is given, so the command can be re-run safely after an interruption.
PostgreSQL authentication is left to ``psql`` itself (a role for your Linux user, see the
runbook); no database password is read or logged here.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fate_icu.utils.checksums import check_manifest

MIMIC_VERSION = "3.1"
MIMIC_CODE_TAG = "v3.0.1"  # built for MIMIC-IV v3.1 (contract.yaml: mimic_code_tag.lab)
MIMIC_CODE_URL = "https://github.com/MIT-LCP/mimic-code.git"
PHYSIONET_URL = f"https://physionet.org/files/mimiciv/{MIMIC_VERSION}/"
EXPECTED_ICU_STAYS = {"3.1": 94_458}  # docs/lab_runbook.md 0.1
MIN_FREE_GB = 150
STEPS = ("preflight", "download", "verify", "load", "concepts", "check")
TOOLS = ("wget", "psql", "createdb", "git", "gzip")
BUILD_SCRIPTS = ("create.sql", "load_gz.sql", "constraint.sql", "index.sql")
Runner = Callable[[Sequence[str], Path | None], int]


@dataclass(frozen=True)
class Settings:
    dest: Path  # where the files are downloaded: holds hosp/ and icu/
    code_dir: Path  # the mimic-code checkout
    db: str = "mimiciv"
    user: str | None = None  # PhysioNet username
    version: str = MIMIC_VERSION
    min_free_gb: int = MIN_FREE_GB
    dry_run: bool = False
    force: bool = False

    @property
    def build_dir(self) -> Path:
        return self.code_dir / "mimic-iv" / "buildmimic" / "postgres"

    @property
    def concepts_dir(self) -> Path:
        return self.code_dir / "mimic-iv" / "concepts_postgres"


def run_command(command: Sequence[str], cwd: Path | None = None) -> int:
    """Run a command with the terminal attached (so wget can ask for a password)."""
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def _run(cfg: Settings, runner: Runner, command: Sequence[str], cwd: Path | None = None) -> None:
    shown = " ".join(str(c) for c in command)
    print(f"$ {shown}", flush=True)
    if cfg.dry_run:
        return
    code = runner(command, cwd)
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {shown}")


def _query_count(cfg: Settings, sql: str) -> int | None:
    """One integer from psql, or None when the table does not exist or the query fails."""
    out = subprocess.run(
        ["psql", "-d", cfg.db, "-At", "-c", sql], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


# ---- steps ------------------------------------------------------------------------------------


def preflight(cfg: Settings, runner: Runner = run_command, which=shutil.which) -> None:
    missing = [tool for tool in TOOLS if which(tool) is None]
    if missing:
        raise SystemExit(f"missing tools: {', '.join(missing)} (see docs/lab_runbook.md, GCP section G4)")
    cfg.dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cfg.dest).free / 1e9
    if free_gb < cfg.min_free_gb:
        raise SystemExit(
            f"only {free_gb:.0f} GB free in {cfg.dest}; MIMIC-IV needs about {cfg.min_free_gb} GB "
            "once loaded: is the data disk mounted?"
        )
    print(f"tools found; {free_gb:.0f} GB free in {cfg.dest}")
    if not cfg.build_dir.is_dir():
        _run(
            cfg,
            runner,
            ["git", "clone", "--branch", MIMIC_CODE_TAG, "--depth", "1", MIMIC_CODE_URL, str(cfg.code_dir)],
        )
    print(f"mimic-code at {cfg.code_dir} ({MIMIC_CODE_TAG})")


def download(cfg: Settings, runner: Runner = run_command) -> None:
    if not cfg.user:
        raise SystemExit("give your PhysioNet username with --user (the password is asked by wget)")
    _run(
        cfg, runner,
        ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-P", str(cfg.dest),
         "--user", cfg.user, "--ask-password", PHYSIONET_URL],
    )  # fmt: skip


def verify(cfg: Settings) -> dict:
    """Check every downloaded file against PhysioNet's checksum list; counts only."""
    if cfg.dry_run:
        print(f"would verify {cfg.dest} against SHA256SUMS.txt")
        return {}
    result = check_manifest(cfg.dest, manifest_name="SHA256SUMS.txt", log=print)
    summary = {k: result[k] for k in ("files_ok", "n_missing", "n_changed", "ok")}
    print(f"verify: {summary}")
    if result["n_missing"] or result["n_changed"]:
        raise SystemExit("some files are missing or damaged: run the download step again (it resumes)")
    return summary


def load(cfg: Settings, runner: Runner = run_command, count=_query_count) -> None:
    if not cfg.force and (count(cfg, "select count(*) from mimiciv_icu.icustays") or 0) > 0:
        print("the database already holds ICU stays: load skipped (use --force to rebuild)")
        return
    if (cfg.dest / "hosp").is_dir() is False or (cfg.dest / "icu").is_dir() is False:
        raise SystemExit(f"{cfg.dest} has no hosp/ and icu/ folders: run the download step first")
    exists = count(cfg, "select 1") is not None
    if not exists:
        _run(cfg, runner, ["createdb", cfg.db])
    data = f"mimic_data_dir={cfg.dest}"
    _run(cfg, runner, ["psql", "-d", cfg.db, "-v", "ON_ERROR_STOP=1", "-f", "create.sql"], cfg.build_dir)
    for script in BUILD_SCRIPTS[1:]:
        _run(
            cfg, runner,
            ["psql", "-d", cfg.db, "-v", "ON_ERROR_STOP=1", "-v", data, "-f", script], cfg.build_dir,
        )  # fmt: skip


def concepts(cfg: Settings, runner: Runner = run_command, count=_query_count) -> None:
    if not cfg.force and (count(cfg, "select count(*) from mimiciv_derived.sofa") or 0) > 0:
        print("mimiciv_derived.sofa already has rows: concepts skipped (use --force to rebuild)")
        return
    for script in ("postgres-functions.sql", "postgres-make-concepts.sql"):
        _run(cfg, runner, ["psql", "-d", cfg.db, "-v", "ON_ERROR_STOP=1", "-f", script], cfg.concepts_dir)


def check(cfg: Settings, count=_query_count) -> dict[str, int]:
    """Row counts only. Fails when a required table is empty or the stay count is not the known one."""
    tables = {
        "icu_stays": "select count(*) from mimiciv_icu.icustays",
        "sofa_rows": "select count(*) from mimiciv_derived.sofa",
        "norepinephrine_equivalent_dose_rows": (
            "select count(*) from mimiciv_derived.norepinephrine_equivalent_dose"
        ),
    }
    if cfg.dry_run:
        print("would count:", ", ".join(tables))
        return {}
    counts = {name: count(cfg, sql) for name, sql in tables.items()}
    print(f"check: {counts}")
    empty = [name for name, n in counts.items() if not n]
    if empty:
        raise SystemExit(f"empty or missing: {', '.join(empty)}")
    expected = EXPECTED_ICU_STAYS.get(cfg.version)
    if expected is not None and counts["icu_stays"] != expected:
        raise SystemExit(
            f"icu_stays is {counts['icu_stays']}, expected {expected} for MIMIC-IV v{cfg.version}"
        )
    return counts  # type: ignore[return-value]


def run_steps(steps: Sequence[str], cfg: Settings, runner: Runner = run_command) -> None:
    for step in steps:
        print(f"\n== {step}", flush=True)
        if step == "preflight":
            preflight(cfg, runner)
        elif step == "download":
            download(cfg, runner)
        elif step == "verify":
            verify(cfg)
        elif step == "load":
            load(cfg, runner)
        elif step == "concepts":
            concepts(cfg, runner)
        elif step == "check":
            check(cfg)
        else:
            raise SystemExit(f"unknown step {step!r}; choose from {', '.join(STEPS)} or all")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", default="all", help=f"one of {', '.join(STEPS)}, or all")
    parser.add_argument("--dest", type=Path, default=Path("/mnt/data/mimic/mimiciv") / MIMIC_VERSION)
    parser.add_argument("--code-dir", type=Path, default=Path("/mnt/data/mimic-code"))
    parser.add_argument("--db", default="mimiciv")
    parser.add_argument("--user", default=None, help="your PhysioNet username")
    parser.add_argument("--min-free-gb", type=int, default=MIN_FREE_GB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild even if the database is already loaded")
    args = parser.parse_args(argv)
    if sys.platform.startswith("win"):
        raise SystemExit(
            "run this on the GCP VM, not on the laptop: credentialed data never goes to the laptop"
        )
    cfg = Settings(
        args.dest,
        args.code_dir,
        args.db,
        args.user,
        MIMIC_VERSION,
        args.min_free_gb,
        args.dry_run,
        args.force,
    )
    run_steps(STEPS if args.step == "all" else [args.step], cfg)


if __name__ == "__main__":
    main()
