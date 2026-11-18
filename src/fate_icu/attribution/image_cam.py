"""Grad-CAM through the frozen DenseNet and the fusion model (Stage 4).

``ImageToLogit`` maps a preprocessed film ``[B, 1, 224, 224]`` to the long-stay logit: the
DenseNet's pooled features and pathology logits become the stay's image vector (exactly as in
``fate_icu.models.image``), and the rest of the stay (time series, static) is held fixed. Grad-CAM
on the last dense block then shows which parts of the film raised (``direction=+1``) or lowered
(``direction=-1``) the logit. Operating thresholds of the image model decide whether a pathology
may be *reported* as a finding at all (``finding_allowed``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fate_icu.attribution.gates import GatedLogit, ones_gates
from fate_icu.models.image import FEATURE_WIDTH, ImageEncoder


class ImageToLogit(nn.Module):
    """Pixels -> long-stay logit for one stay, with everything else fixed."""

    def __init__(self, encoder: ImageEncoder, gated: GatedLogit, fixed_stay: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.encoder, self.gated = encoder, gated
        self.fixed = {k: v[:1] for k, v in fixed_stay.items()}
        for module in encoder.model.modules():  # in-place ReLUs break gradient hooks
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def vector(self, x224: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.model.features2(x224)
        logits = self.encoder.model.classifier(pooled)[:, self.encoder.pathology_index]
        return torch.cat([pooled, logits], dim=1)

    def forward(self, x224: torch.Tensor) -> torch.Tensor:
        n = x224.shape[0]
        f = {k: v.expand(n, *v.shape[1:]) for k, v in self.fixed.items()}
        g = ones_gates(self.gated.layout, n)
        present = torch.ones(n, dtype=torch.bool)
        return self.gated(
            g["ts"],
            g["static"],
            g["concept"],
            present,
            f["values"],
            f["mask"],
            f["delta"],
            f["static"],
            self.vector(x224),
        )


class _Signed:
    """Grad-CAM target: the logit, or minus the logit to see what pulled it down."""

    def __init__(self, sign: float) -> None:
        self.sign = sign

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.sign * model_output


def grad_cam(wrapper: ImageToLogit, x224: np.ndarray, direction: int = 1) -> np.ndarray:
    """Heatmap ``[224, 224]`` in [0, 1] of what pushed the logit up (+1) or down (-1)."""
    from pytorch_grad_cam import GradCAM

    if direction not in (1, -1):
        raise ValueError("direction must be +1 (toward longer) or -1 (toward shorter)")
    x = torch.as_tensor(np.asarray(x224), dtype=torch.float32)
    if x.ndim == 3:
        x = x[None]
    x = x.clone().requires_grad_(True)  # the weights are frozen: the film starts the graph
    layer = wrapper.encoder.model.features.denseblock4
    with GradCAM(model=wrapper, target_layers=[layer]) as cam:
        heat = cam(input_tensor=x, targets=[_Signed(float(direction))])
    return heat[0].astype(np.float32)


def operating_thresholds(encoder: ImageEncoder) -> dict[str, float]:
    """Each concept's operating threshold on the model's sigmoid output (NaN when it has none)."""
    thresholds = getattr(encoder.model, "op_threshs", None)
    out = {}
    for index, name in zip(encoder.pathology_index, encoder.pathology_names, strict=True):
        value = float("nan") if thresholds is None else float(thresholds[index])
        out[name] = value
    return out


def concept_probabilities(
    vector: np.ndarray, concepts: list[str], width: int = FEATURE_WIDTH
) -> dict[str, float]:
    """The image model's probability of each concept: the sigmoid of its raw logit."""
    logits = np.asarray(vector, dtype=np.float64)[width : width + len(concepts)]
    return {name: float(1.0 / (1.0 + np.exp(-z))) for name, z in zip(concepts, logits, strict=True)}


def finding_allowed(probability: float, threshold: float) -> bool:
    """A radiographic finding may be reported only above the model's own operating threshold.

    A concept without a threshold (NaN) is never reported as a finding.
    """
    return bool(np.isfinite(threshold) and probability > threshold)
