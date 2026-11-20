"""Copy the selected MIMIC-CXR-JPG images from PhysioNet's Google Cloud bucket (Stage 1, D-052).

    python -m fate_icu.data.cxr.gcs_fetch --link-run <run id> --billing-project <project id>

Runs on the GCP VM (or the lab) only, never on the laptop (CLAUDE.md A3). Stage 1's link step writes
``image_list.txt``, the relative path of every selected image (``files/p10/p10000032/s.../<id>.jpg``).
This copies exactly those files from ``gs://mimic-cxr-jpg-2.1.0.physionet.org`` to ``MIMIC_CXR_DIR`` with
the same folder layout, so the cache step (``fate_icu.data.cxr.cache``) reads them where it expects.
It replaces the slow ``wget`` of runbook 1.3: from the bucket to a VM in Google's network it takes minutes.

How the access works (PhysioNet grants it to your *Google account*):

* the VM must be signed in as that account: ``gcloud auth login --no-launch-browser`` on the VM, and
  ``gcloud auth revoke`` when the copy is done. No account name is read or logged here;
* PhysioNet allows reading a file by its exact name but not listing the bucket, which is why the
  exact list from the link step is used and no wildcard;
* the bucket is "requester pays", so every command carries ``--billing-project`` (your project id).

Resumable: a file that is already there is skipped (``gcloud`` writes to a temporary name and renames
when complete, so a present file is a finished one) and it checks each file's hash. The copies run in
parallel (``--workers``); the result does not depend on their order (tests/test_cxr_gcs_fetch.py).
Prints counts only; the paths of failures go to ``problems.txt`` in the run folder, never to the screen.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from fate_icu.data.cxr.link import IMAGE_LIST_FILE
from fate_icu.utils.aggregates import format_aggregates, suppress_counts
from fate_icu.utils.env import Settings, load_settings
from fate_icu.utils.manifest import new_run_id

BUCKET = "mimic-cxr-jpg-2.1.0.physionet.org"
METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
ALLOWED_MACHINES = ("gcp", "lab")
Runner = Callable[[Sequence[str]], tuple[int, str]]  # (exit code, last line of the error text)


@dataclass(frozen=True)
class FetchSettings:
    billing_project: str  # the project that pays for the requester-pays reads
    bucket: str = BUCKET
    workers: int = 16
    retries: int = 3
    gcloud: str = "gcloud"
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.billing_project.strip():
            raise ValueError("a requester-pays bucket needs --billing-project (your GCP project id)")
        if self.workers < 1 or self.retries < 1:
            raise ValueError("workers and retries must be at least 1")


@dataclass
class FetchResult:
    requested: int = 0
    already_present: int = 0
    copied: int = 0
    failed: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "already_present": self.already_present,
            "copied_now": self.copied,
            "failed": len(self.failed),
        }


def valid_relative(path: str) -> bool:
    """A path from the link step: ``files/...jpg``, relative, with no way out of the destination."""
    parts = PurePosixPath(path).parts
    return (
        path.startswith("files/")
        and path.endswith(".jpg")
        and "\\" not in path
        and "//" not in path  # PurePosixPath would quietly tidy it
        and ".." not in parts
        and not PurePosixPath(path).is_absolute()
        and all(parts)
    )


def read_list(path: Path) -> list[str]:
    """The unique, sorted image paths of a list file; any path that could leave the folder is an error."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [line for line in lines if not valid_relative(line)]
    if bad:
        raise ValueError(f"{len(bad)} line(s) of {Path(path).name} are not image paths under files/")
    return sorted(set(lines))


def run_gcloud(command: Sequence[str]) -> tuple[int, str]:
    out = subprocess.run(list(command), capture_output=True, text=True, check=False)
    last = (out.stderr.strip().splitlines() or [""])[-1][:200]
    return out.returncode, last


def copy_command(cfg: FetchSettings, source: str, destination: Path) -> list[str]:
    return [
        cfg.gcloud, "storage", "cp", f"--billing-project={cfg.billing_project}",
        f"gs://{cfg.bucket}/{source}", str(destination),
    ]  # fmt: skip


def copy_one(relative: str, dest_root: Path, cfg: FetchSettings, runner: Runner) -> str:
    """``present``, ``copied`` or ``failed`` for one image."""
    target = dest_root / relative
    if target.is_file() and target.stat().st_size > 0:
        return "present"
    target.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(cfg.retries):
        code, _message = runner(copy_command(cfg, relative, target))
        if code == 0:
            return "copied"
    return "failed"


def fetch(
    relatives: Sequence[str],
    dest_root: Path,
    cfg: FetchSettings,
    runner: Runner = run_gcloud,
    log: Callable[[str], None] = print,
) -> FetchResult:
    """Copy every image in ``relatives`` that is not already in ``dest_root``."""
    result = FetchResult(requested=len(relatives))
    if cfg.dry_run:
        template = (
            f"{cfg.gcloud} storage cp --billing-project={cfg.billing_project} "
            f"gs://{cfg.bucket}/<image path> {dest_root}/<image path>"
        )
        log(f"would copy up to {len(relatives)} images with: {template}")
        return result
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        outcomes = list(pool.map(lambda rel: copy_one(rel, dest_root, cfg, runner), relatives))
    for relative, outcome in zip(
        relatives, outcomes, strict=True
    ):  # in list order, whatever the thread order
        if outcome == "present":
            result.already_present += 1
        elif outcome == "copied":
            result.copied += 1
        else:
            result.failed.append(relative)
    log(f"images: {result.counts()}")
    return result


def fetch_metadata(dest_root: Path, cfg: FetchSettings, runner: Runner = run_gcloud) -> bool:
    """The small metadata file the link step reads (about 16 MB); True when it is there afterwards."""
    target = dest_root / METADATA_FILE
    if target.is_file() and target.stat().st_size > 0:
        return True
    if cfg.dry_run:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    return runner(copy_command(cfg, METADATA_FILE, target))[0] == 0


def check_machine(settings: Settings) -> None:
    if settings.machine not in ALLOWED_MACHINES:
        raise SystemExit(
            f"FATE_MACHINE is {settings.machine!r}: credentialed images are copied only on the GCP VM or "
            "the lab (CLAUDE.md A3), never on the laptop"
        )


def resolve_list(settings: Settings, link_run: str | None, list_file: Path | None) -> Path:
    if bool(link_run) == bool(list_file):
        raise SystemExit("give exactly one of --link-run and --list")
    if list_file:
        return Path(list_file)
    folder = Path(link_run) if Path(link_run).is_dir() else settings.artifacts_dir / "cxr" / link_run
    path = folder / IMAGE_LIST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {IMAGE_LIST_FILE} in {folder}: run the Stage 1 link step first")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--link-run", default=None, help="Stage 1 link run id (or folder) holding image_list.txt"
    )
    parser.add_argument("--list", type=Path, default=None, help="a list file instead of --link-run")
    parser.add_argument("--dest", type=Path, default=None, help="default: MIMIC_CXR_DIR from .env")
    parser.add_argument("--billing-project", required=True, help="your GCP project id (requester pays)")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--metadata", action="store_true", help="also copy mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform.startswith("win"):
        raise SystemExit(
            "run this on the GCP VM, not on the laptop: credentialed data never goes to the laptop"
        )
    settings = load_settings()
    check_machine(settings)
    dest = args.dest or settings.mimic_cxr_dir
    if dest is None:
        raise SystemExit("MIMIC_CXR_DIR is not set (see .env.example) and --dest was not given")
    cfg = FetchSettings(args.billing_project, args.bucket, args.workers, dry_run=args.dry_run)
    report: dict = {}
    if args.metadata:
        report["metadata_present"] = fetch_metadata(dest, cfg)
    if args.link_run or args.list:
        relatives = read_list(resolve_list(settings, args.link_run, args.list))
        result = fetch(relatives, dest, cfg)
        report["images"] = suppress_counts(result.counts())
        if result.failed:
            run_dir = settings.artifacts_dir / "cxr_fetch_runs" / new_run_id("cxr_fetch")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "problems.txt").write_text("".join(f"{p}\n" for p in result.failed), encoding="utf-8")
            report["problems_file"] = str(run_dir / "problems.txt")
            print(format_aggregates(report))
            raise SystemExit("some images could not be copied: is the VM signed in as the approved account?")
    print(format_aggregates(report))


if __name__ == "__main__":
    main()
