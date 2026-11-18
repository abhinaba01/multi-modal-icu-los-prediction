"""Frozen chest X-ray encoder: torchxrayvision DenseNet-121, weights ``densenet121-res224-chex``.

The encoder is never trained (Stage 3). ``chex`` was not trained on MIMIC-CXR, so it has not
seen our test images; ``-all`` is an ablation only. For each preprocessed image (Stage 1,
``x224``: float32 ``[1, 224, 224]``) the feature vector is

    [ 1024-d pooled DenseNet features  ||  logits of the non-empty pathology outputs ]

(1024 + 11 = 1035 values for ``chex``). The logits are the raw classifier outputs, taken
before the sigmoid and the operating-point normalisation that ``model(x)`` applies.

Features are cached per image key (the ``dicom_id``), keyed by the weights name, so a cache
made with other weights is never reused. ``tests/test_models_image.py`` checks that cached
and on-the-fly features are identical.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fate_icu.data.cxr.preprocess import _xrv, load_cached

WEIGHTS = "densenet121-res224-chex"
FEATURE_WIDTH = 1024


@dataclass
class ImageEncoder:
    """The frozen network and the indices of its non-empty pathology outputs."""

    model: torch.nn.Module
    weights: str
    pathology_index: list[int]
    pathology_names: list[str]

    @property
    def dim(self) -> int:
        return FEATURE_WIDTH + len(self.pathology_index)

    @torch.no_grad()
    def encode(self, x224: np.ndarray | torch.Tensor, batch_size: int = 16) -> np.ndarray:
        """Feature vectors ``[N, dim]`` (float32) for images ``[N, 1, 224, 224]``."""
        x = torch.as_tensor(np.asarray(x224), dtype=torch.float32)
        if x.ndim == 3:
            x = x[None]
        if x.ndim != 4 or x.shape[1:] != (1, 224, 224):
            raise ValueError(f"expected images of shape [N, 1, 224, 224], got {tuple(x.shape)}")
        out = []
        for start in range(0, len(x), batch_size):
            pooled = self.model.features2(x[start : start + batch_size])
            logits = self.model.classifier(pooled)[:, self.pathology_index]
            out.append(torch.cat([pooled, logits], dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


def load_image_encoder(weights: str = WEIGHTS) -> ImageEncoder:
    """Load (and download once, ~28 MB for ``chex``) the frozen torchxrayvision DenseNet-121."""
    xrv = _xrv()
    with warnings.catch_warnings():
        # torchxrayvision pickles whole modules; torch warns that their source differs. Only that
        # known message is hidden here.
        source_change = getattr(torch.serialization, "SourceChangeWarning", Warning)
        warnings.filterwarnings("ignore", category=source_change)
        model = xrv.models.DenseNet(weights=weights).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    index = [i for i, name in enumerate(model.pathologies) if name]
    names = [model.pathologies[i] for i in index]
    return ImageEncoder(model, weights, index, names)


def feature_path(feature_dir: str | Path, weights: str, key: str) -> Path:
    key = str(key)
    return Path(feature_dir) / weights / key[:2] / f"{key}.npy"


def cached_features(
    keys: Iterable[str],
    image_cache: str | Path,
    feature_dir: str | Path,
    encoder: ImageEncoder,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Feature vector per key, computing and storing the ones not cached yet.

    Args:
        keys: Image keys (``dicom_id``).
        image_cache: The Stage 1 preprocessed-image cache.
        feature_dir: Root of the feature cache (one ``.npy`` per key and weights name).
        encoder: The frozen encoder.
        batch_size: Images per forward pass.

    Raises:
        FileNotFoundError: A key has no preprocessed image (fail loudly, CLAUDE.md A4).
    """
    keys = [str(k) for k in dict.fromkeys(keys)]
    result: dict[str, np.ndarray] = {}
    todo = []
    for key in keys:
        path = feature_path(feature_dir, encoder.weights, key)
        if path.exists():
            result[key] = np.load(path, allow_pickle=False)
        else:
            todo.append(key)
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        images = np.stack([load_cached(image_cache, key)["x224"] for key in chunk])
        for key, vector in zip(chunk, encoder.encode(images, batch_size), strict=True):
            path = feature_path(feature_dir, encoder.weights, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, vector, allow_pickle=False)
            tmp.replace(path)
            result[key] = vector
    return result


def main() -> None:
    """Download the weights once (about 28 MB) and print the feature layout."""
    encoder = load_image_encoder()
    print(
        f"{encoder.weights}: {FEATURE_WIDTH} pooled features + {len(encoder.pathology_index)} "
        f"pathology logits = {encoder.dim} values per image"
    )


if __name__ == "__main__":
    main()
