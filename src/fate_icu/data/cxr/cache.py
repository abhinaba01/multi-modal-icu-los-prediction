"""Preprocess the selected chest X-rays into the cache (Stage 1).

    python -m fate_icu.data.cxr.cache --config configs/cxr/smoke.yaml                  # public test images
    python -m fate_icu.data.cxr.cache --config configs/cxr/full.yaml --link-run <run>  # lab: selected images

The cache lives in ``<artifacts>/cxr_cache/`` (one ``.npz`` per image, resumable). Each
run writes a manifest and a summary under ``<artifacts>/cxr_cache_runs/<run_id>/``.
Missing or unreadable images are counted; their keys go to ``problems.txt`` in that run
folder, never to the printout. The command exits with an error if any image is missing or
unreadable (CLAUDE.md A4, fail loudly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fate_icu.data.cxr.link import PAIRED_FILE, CxrConfig, load_config
from fate_icu.data.cxr.linkage import relative_path
from fate_icu.data.cxr.preprocess import build_cache
from fate_icu.utils.aggregates import format_aggregates, suppress_counts
from fate_icu.utils.env import REPO_ROOT, Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def fixture_items(folder: Path) -> list[tuple[str, Path]]:
    """``(file stem, path)`` for the public test images."""
    return [(p.stem, p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]


def mimic_items(paired: pd.DataFrame, root: Path) -> list[tuple[str, Path]]:
    """``(dicom_id, path)`` for the selected MIMIC-CXR images."""
    linked = paired[paired["has_cxr"]].drop_duplicates("dicom_id")
    return [
        (d, root / relative_path(s, st, d))
        for s, st, d in zip(linked["subject_id"], linked["study_id"], linked["dicom_id"], strict=True)
    ]


def run(config: CxrConfig, link_run: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    run_id = new_run_id(f"{config.run_name}_cache")
    run_dir = settings.artifacts_dir / "cxr_cache_runs" / run_id
    cache_dir = settings.artifacts_dir / "cxr_cache"
    if config.images == "fixtures":
        items = fixture_items(REPO_ROOT / config.fixtures_dir)
    else:
        if link_run is None:
            raise RuntimeError(
                "--link-run is required for MIMIC images (the run id or folder of the link step)"
            )
        if settings.mimic_cxr_dir is None:
            raise RuntimeError("MIMIC_CXR_DIR is not set (see .env.example)")
        link_dir = Path(link_run) if Path(link_run).is_dir() else settings.artifacts_dir / "cxr" / link_run
        items = mimic_items(pd.read_parquet(link_dir / PAIRED_FILE), settings.mimic_cxr_dir)
    summary = build_cache(items, cache_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "problems.txt").write_text(
        "".join(f"missing\t{k}\n" for k in summary.missing)
        + "".join(f"failed\t{k}\n" for k in summary.failed),
        encoding="utf-8",
    )
    report = {
        "run_id": run_id,
        "images": config.images,
        "counts": suppress_counts(summary.counts()),
        "cache_dir": str(cache_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_run_manifest(
        run_dir,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        data={"images": config.images, "link_run": link_run},
        extra={"machine": settings.machine},
    )
    report["ok"] = not summary.missing and not summary.failed
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--link-run", default=None, help="run id or folder of the link step (MIMIC images)")
    args = parser.parse_args(argv)
    config = load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    report = run(config, args.link_run)
    print(format_aggregates(report))
    if not report["ok"]:
        raise SystemExit("some images are missing or unreadable; see problems.txt in the cache run folder")


if __name__ == "__main__":
    main()
