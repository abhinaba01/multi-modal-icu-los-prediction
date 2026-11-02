"""Chest X-ray preprocessing for the image encoder, following torchxrayvision.

1. read the file to a 2-D grayscale array: colour channels are averaged (as the
   torchxrayvision README does), an alpha channel is dropped, and the maximum value comes
   from the bit depth (255 for 8-bit, 65535 for 16-bit);
2. ``xrv.datasets.normalize`` to [-1024, 1024];
3. ``XRayCenterCrop`` (the central square on the shorter side);
4. ``XRayResizer(224)`` for the model (float32 ``[1, 224, 224]``) and ``XRayResizer(512)``
   for Grad-CAM display (stored as 8-bit ``[512, 512]``);
5. cache one compressed ``.npz`` per image key (the ``dicom_id`` for MIMIC-CXR).

torchxrayvision silences every Python warning when it is imported (it calls
``warnings.filterwarnings("ignore")`` at import time). It is imported here inside
``warnings.catch_warnings()`` so the filters are restored and our own warnings still show.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

MODEL_SIZE, DISPLAY_SIZE = 224, 512
CACHE_KEYS = ("x224", "display512")


@lru_cache(maxsize=1)
def _xrv() -> ModuleType:
    """torchxrayvision, imported without changing the global warning filters."""
    with warnings.catch_warnings():
        import torchxrayvision as xrv
    return xrv


def read_xray(path: str | Path) -> tuple[np.ndarray, int]:
    """A 2-D grayscale array and its maximum possible value.

    Raises:
        ValueError: An image mode that is not 8-bit or 16-bit grayscale or colour.
    """
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
            return np.asarray(image, dtype=np.uint16), 65535
        if mode == "I":
            array = np.asarray(image)
            if array.min() < 0 or array.max() > 65535:
                raise ValueError(f"{path}: 32-bit integer image outside the 16-bit range")
            return array.astype(np.uint16), 65535
        if mode in ("L", "1"):
            return np.asarray(image.convert("L"), dtype=np.uint8), 255
        if mode in ("RGB", "RGBA", "LA", "P", "CMYK", "YCbCr"):
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            return rgb.mean(axis=2), 255
    raise ValueError(f"{path}: unsupported image mode {mode!r}")


def preprocess_xray(array: np.ndarray, maxval: int) -> dict[str, np.ndarray]:
    """``x224`` (float32 ``[1, 224, 224]`` in [-1024, 1024]) and ``display512`` (uint8)."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {array.shape}")
    xrv = _xrv()
    image = xrv.datasets.normalize(array, maxval)[None, :, :]
    image = xrv.datasets.XRayCenterCrop()(image)
    x224 = xrv.datasets.XRayResizer(MODEL_SIZE)(image).astype(np.float32)
    x512 = xrv.datasets.XRayResizer(DISPLAY_SIZE)(image)
    display = np.clip(np.round((x512[0] + 1024.0) / 2048.0 * 255.0), 0, 255).astype(np.uint8)
    return {"x224": x224, "display512": display}


def cache_path(cache_dir: str | Path, key: str) -> Path:
    """``<cache>/<first two characters of the key>/<key>.npz``."""
    key = str(key)
    return Path(cache_dir) / key[:2] / f"{key}.npz"


def load_cached(cache_dir: str | Path, key: str) -> dict[str, np.ndarray]:
    with np.load(cache_path(cache_dir, key), allow_pickle=False) as data:
        return {name: data[name] for name in CACHE_KEYS}


@dataclass
class CacheSummary:
    """Counts of one cache build; ``missing``/``failed`` keys are for a local file only."""

    requested: int = 0
    cached_now: int = 0
    already_cached: int = 0
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "cached_now": self.cached_now,
            "already_cached": self.already_cached,
            "missing_files": len(self.missing),
            "failed_to_read": len(self.failed),
        }


def build_cache(
    items: Iterable[tuple[str, Path]], cache_dir: str | Path, overwrite: bool = False
) -> CacheSummary:
    """Preprocess and cache images; images already cached are skipped (resumable).

    Args:
        items: ``(key, path)`` pairs, the key being the ``dicom_id`` for MIMIC-CXR.
        cache_dir: Cache root.
        overwrite: Recompute images that are already cached.
    """
    summary = CacheSummary()
    for key, path in items:
        summary.requested += 1
        target = cache_path(cache_dir, key)
        if target.exists() and not overwrite:
            summary.already_cached += 1
            continue
        if not Path(path).exists():
            summary.missing.append(str(key))
            continue
        try:
            arrays = preprocess_xray(*read_xray(path))
        except (OSError, ValueError):
            summary.failed.append(str(key))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, target)  # never leave a half-written cache file
        summary.cached_now += 1
    return summary
