"""Collect what the Kaggle or Colab GPU step needs into one folder (laptop).

    python -m fate_icu.rag.gpu_bundle --config configs/rag/full.yaml

Builds a fresh fate_icu wheel from the current code (``uv build --wheel``) and copies it,
the corpus (``passages.parquet``), ``gpu_spec.json`` and ``bm25_hits.json`` into
``<FATE_RAG_DIR>/<run_name>/gpu_bundle/``. Upload those four files as one private Kaggle
dataset (or to Google Drive for Colab) and run the printed commands
(docs/lab_runbook.md, Stage 5). Everything in the bundle is public data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from fate_icu.rag.build_index import read_index_info
from fate_icu.rag.config import RagConfig, load_config, rag_paths
from fate_icu.rag.corpus import PASSAGES_FILE
from fate_icu.rag.gpu_step import GPU_OUT_DIR, GPU_SPEC_FILE
from fate_icu.rag.query_bank import BM25_HITS_FILE
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.env import REPO_ROOT, Settings, load_settings

# Pinned on the GPU machine so its encoder and index code match the laptop's.
GPU_PACKAGES = ("transformers", "tokenizers", "huggingface-hub", "safetensors", "faiss-cpu", "pydantic")
DATASET_DIR = "/kaggle/input/<your-dataset-name>"


def pip_pins() -> list[str]:
    return [f"{name}=={metadata.version(name)}" for name in GPU_PACKAGES]


def build_wheel(out_dir: Path) -> Path:
    """A wheel of the current source tree (fails loudly without uv)."""
    for old in out_dir.glob("fate_icu-*.whl"):
        old.unlink()
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("uv is not on PATH: open a new terminal after installing uv") from exc
    wheels = sorted(out_dir.glob("fate_icu-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel in {out_dir}, found {len(wheels)}")
    return wheels[0]


def run(config: RagConfig, *, settings: Settings | None = None, wheel_builder=build_wheel) -> dict[str, Any]:
    settings = settings or load_settings()
    paths = rag_paths(config, settings)
    version = paths.current("index")
    index_dir = paths.index(version)
    info = read_index_info(index_dir)
    bundle = paths.build / "gpu_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    wheel = wheel_builder(bundle)
    shutil.copy2(paths.corpus(info["corpus_id"]) / PASSAGES_FILE, bundle / PASSAGES_FILE)
    for name in (GPU_SPEC_FILE, BM25_HITS_FILE):
        shutil.copy2(index_dir / name, bundle / name)
    files = {p.name: f"{p.stat().st_size / 1e6:,.1f} MB" for p in sorted(bundle.iterdir()) if p.is_file()}
    commands = [
        "pip install -q " + " ".join(pip_pins()),
        f"pip install -q --no-deps --ignore-requires-python {DATASET_DIR}/{wheel.name}",
        f"python -m fate_icu.rag.gpu_step --bundle {DATASET_DIR} "
        f"--out /kaggle/working/{GPU_OUT_DIR} --device cuda",
    ]
    return {
        "bundle_dir": str(bundle),
        "files": files,
        "index_version": version,
        "passages": info["n_passages"],
        "notebook_commands": commands,
        "then": f"download /kaggle/working/{GPU_OUT_DIR} into {index_dir / GPU_OUT_DIR}, "
        "then run build_index --part dense",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/rag/full.yaml")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config))))


if __name__ == "__main__":
    main()
