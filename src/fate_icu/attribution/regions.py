"""Where on the film an image attribution sits, in anatomical words (Stage 4).

Description only: it never decides which units are drivers. torchxrayvision's ChestX-Det
PSPNet segments 14 structures on the 512 x 512 display copy of Stage 1. Its labels are
*anatomical*: "Right Lung" is the patient's right lung, which on a frontal (PA or AP) film is
on the image's left. Sides therefore come from the label, never from image columns
(``tests/test_attr_regions.py`` checks this on public films). Each lung is split at the middle
of its own vertical extent into an upper and a lower zone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fate_icu.data.cxr.preprocess import _xrv

SEGMENTATION_SIZE = 512
ZONES = ("right upper lung zone", "right lower lung zone", "left upper lung zone", "left lower lung zone")
CARDIOMEDIASTINAL = "cardiomediastinal region"
OTHER = "outside the lung fields and heart"
DIFFUSE = "no single anatomical region"


@dataclass
class Segmenter:
    """The anatomical segmentation network (weights download once, about 273 MB)."""

    model: torch.nn.Module

    @property
    def targets(self) -> list[str]:
        return list(self.model.targets)

    @torch.no_grad()
    def masks(self, display512: np.ndarray) -> dict[str, np.ndarray]:
        """Bool masks ``[512, 512]`` by structure name from the stored 8-bit display image."""
        if display512.shape != (SEGMENTATION_SIZE, SEGMENTATION_SIZE):
            raise ValueError(
                f"expected a {SEGMENTATION_SIZE}x{SEGMENTATION_SIZE} image, got {display512.shape}"
            )
        x = display512.astype(np.float32) / 255.0 * 2048.0 - 1024.0  # back to the model's scale
        prob = torch.sigmoid(self.model(torch.from_numpy(x)[None, None]))[0].numpy()
        return {name: prob[i] > 0.5 for i, name in enumerate(self.targets)}


def load_segmenter() -> Segmenter:
    xrv = _xrv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = xrv.baseline_models.chestx_det.PSPNet().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Segmenter(model)


def lung_zones(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Upper and lower zone of each lung, from the lung's own vertical extent.

    A lung that the network did not find gives empty zones (the region is then undescribed).
    """
    zones: dict[str, np.ndarray] = {}
    for side, label in (("right", "Right Lung"), ("left", "Left Lung")):
        lung = masks[label]
        upper = np.zeros_like(lung)
        lower = np.zeros_like(lung)
        rows = np.flatnonzero(lung.any(axis=1))
        if len(rows):
            middle = (rows[0] + rows[-1]) / 2.0
            row_index = np.arange(lung.shape[0])[:, None]
            upper = lung & (row_index <= middle)
            lower = lung & (row_index > middle)
        zones[f"{side} upper lung zone"] = upper
        zones[f"{side} lower lung zone"] = lower
    return zones


def region_shares(heatmap: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of the heatmap's mass in each zone, in the cardiomediastinal region and elsewhere."""
    heat = np.asarray(heatmap, dtype=np.float64)
    if heat.shape != (SEGMENTATION_SIZE, SEGMENTATION_SIZE):
        heat = F.interpolate(
            torch.from_numpy(heat)[None, None].float(), size=(SEGMENTATION_SIZE,) * 2, mode="bilinear",
            align_corners=False,
        )[0, 0].numpy().astype(np.float64)  # fmt: skip
    heat = np.clip(heat, 0.0, None)
    total = heat.sum()
    if total <= 0:
        return {}
    zones = lung_zones(masks)
    taken = np.zeros_like(masks["Heart"])
    shares: dict[str, float] = {}
    for name in ZONES:
        shares[name] = float(heat[zones[name]].sum() / total)
        taken |= zones[name]
    middle = (masks["Heart"] | masks["Mediastinum"]) & ~taken
    shares[CARDIOMEDIASTINAL] = float(heat[middle].sum() / total)
    taken |= middle
    shares[OTHER] = float(heat[~taken].sum() / total)
    return shares


def describe_region(shares: dict[str, float], min_share: float = 0.4) -> str:
    """The lung zone or cardiomediastinal region holding at least ``min_share`` of the mass.

    The remainder (outside the lungs and heart) is never named: it is most of the picture, so a
    scattered map would put most of its mass there by area alone. Such a map, and one split
    between regions, is described as ``DIFFUSE``.
    """
    anatomical = {name: shares[name] for name in (*ZONES, CARDIOMEDIASTINAL) if name in shares}
    if not anatomical:
        return DIFFUSE
    name = max(anatomical, key=lambda k: (anatomical[k], k))
    return name if anatomical[name] >= min_share else DIFFUSE
