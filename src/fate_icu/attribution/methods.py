"""Attribution methods over evidence units (Stage 4).

Every method scores the units of ``UnitSpace`` for a batch of stays and explains the long-stay
logit, holding the other evidence fixed through captum's ``additional_forward_args``. Scores
are in the unit order of the space. Two kinds:

* **signed** (``occlusion``, ``ablation``, ``ig``): positive pushes the logit up, toward a
  longer stay; occlusion and ablation are the exact drop ``logit(x) - logit(x without unit)``;
* **importance** (``extremal_mask``, ``dynamask``): how much the prediction relies on the unit,
  from 0 to 1, with no direction. The direction and the effect size of a chosen driver always
  come from the removal Δ of ``select.py``.

Integrated Gradients runs on the gates, with the baseline "everything removed" (gate 0): for
time series that is the training-time "never observed" input, for static features the
"missing" representation (D-048). Time-series scores are computed per hour and then summed (IG)
or averaged (masks) over each 4-hour block.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from fate_icu.attribution.gates import GatedLogit, ones_gates
from fate_icu.attribution.units import UnitSpace

TS_METHODS = ("occlusion", "ig", "extremal_mask", "dynamask")
STATIC_METHODS = ("ablation", "ig")
IMAGE_METHODS = ("ablation", "ig")
SIGNED = {"occlusion", "ablation", "ig"}


@dataclass(frozen=True)
class MethodConfig:
    """Settings of the methods (the smoke config uses small numbers)."""

    ig_steps: int = 32
    mask_epochs: int = 500
    mask_lr: float = 0.01
    mask_lambda_1: float = 3.0  # L1 weight on the mask (tuned on a separate seed, D-048)
    mask_batch_size: int = 32
    dynamask_epochs: int = 200
    seed: int = 666
    extra: dict = field(default_factory=dict)


class BatchScorer:
    """Logits of a batch of stays, each with its own set of removed units."""

    def __init__(self, model: GatedLogit, space: UnitSpace, fixed: dict[str, torch.Tensor]) -> None:
        self.model, self.space, self.fixed = model, space, fixed
        self.n = fixed["values"].shape[0]

    @torch.no_grad()
    def logits(self, removed: np.ndarray) -> np.ndarray:
        """``removed``: bool ``[N, n_units]``, one removal mask per stay."""
        gates = self.space.gates(removed)
        f = self.fixed
        present = f["image_present"] & ~gates["image_removed"]
        return self.model(
            gates["ts"], gates["static"], gates["concept"], present,
            f["values"], f["mask"], f["delta"], f["static"], f["image"],
        ).numpy()  # fmt: skip

    def base(self) -> np.ndarray:
        return self.logits(np.zeros((self.n, len(self.space)), dtype=bool))

    def delta(self, units: np.ndarray | None = None) -> np.ndarray:
        """Exact Δ per stay and unit, ``[N, len(units)]``: ``logit(x) - logit(x without unit)``."""
        units = np.arange(len(self.space)) if units is None else np.asarray(units)
        base = self.base()
        out = np.zeros((self.n, len(units)), dtype=np.float32)
        for j, u in enumerate(units):
            removed = np.zeros((self.n, len(self.space)), dtype=bool)
            removed[:, u] = True
            out[:, j] = base - self.logits(removed)
        return out


def _extra(fixed: dict[str, torch.Tensor], space: UnitSpace, keep: str) -> tuple:
    """The fixed arguments after the attributed gate: the other two gates, then the stay tensors."""
    n = fixed["values"].shape[0]
    g = ones_gates(space.layout, n)
    others = {
        "ts": (g["static"], g["concept"]),
        "static": (g["ts"], g["concept"]),
        "concept": (g["ts"], g["static"]),
    }
    stay = (
        fixed["image_present"],
        fixed["values"],
        fixed["mask"],
        fixed["delta"],
        fixed["static"],
        fixed["image"],
    )
    return (*others[keep], *stay)


def _forward(model: GatedLogit, keep: str) -> Callable:
    """``f(gate, *fixed)`` where the attributed gate comes first."""
    if keep == "ts":
        return lambda gate, sg, cg, ip, v, m, d, s, im: model(gate, sg, cg, ip, v, m, d, s, im)
    if keep == "static":
        return lambda gate, tg, cg, ip, v, m, d, s, im: model(tg, gate, cg, ip, v, m, d, s, im)
    return lambda gate, tg, sg, ip, v, m, d, s, im: model(tg, sg, gate, ip, v, m, d, s, im)


def _ig(
    model: GatedLogit, space: UnitSpace, fixed: dict, keep: str, gate: torch.Tensor, steps: int
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    attr = IntegratedGradients(_forward(model, keep), multiply_by_inputs=True).attribute(
        gate,
        baselines=torch.zeros_like(gate),
        additional_forward_args=_extra(fixed, space, keep),
        n_steps=steps,
    )
    return attr.detach().numpy()


# ---- time series -------------------------------------------------------------------------


def ts_occlusion(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    """Exact Δ of every (variable, block) unit: ``[N, n_ts]``."""
    return BatchScorer(model, space, fixed).delta(np.arange(space.n_ts))


def ts_ig(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    n = fixed["values"].shape[0]
    hourly = _ig(model, space, fixed, "ts", ones_gates(space.layout, n)["ts"], cfg.ig_steps)
    return np.stack([space.blocks_to_units(space.hourly_to_blocks(h)) for h in hourly])


def _lightning_trainer(epochs: int):
    from pytorch_lightning import Trainer

    for name in ("pytorch_lightning", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, deterministic=False,
    )  # fmt: skip


def _block_mean(space: UnitSpace, hourly: np.ndarray) -> np.ndarray:
    blocks = np.stack([hourly[:, s:e, :].mean(axis=1) for s, e in space.blocks], axis=1)  # [N, n_blocks, C]
    return np.stack([space.blocks_to_units(b) for b in blocks])


def ts_extremal_mask(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    """tint's ExtremalMask on the time-series gates: a mask learned to keep the logit while
    dropping as much as possible. Importance in [0, 1] per unit, ``[N, n_ts]``."""
    from tint.attr import ExtremalMask
    from tint.attr.models import ExtremalMaskNet

    torch.manual_seed(cfg.seed)
    n = fixed["values"].shape[0]
    inner = _forward(model, "ts")

    def f(gate: torch.Tensor, *rest) -> torch.Tensor:
        return inner(gate, *rest).unsqueeze(-1)  # tint wants [B, 1]

    gate = ones_gates(space.layout, n)["ts"]
    net = ExtremalMaskNet(
        forward_func=f, preservation_mode=True, batch_size=min(cfg.mask_batch_size, n),
        lambda_1=cfg.mask_lambda_1, lr=cfg.mask_lr,
    )  # fmt: skip
    hourly = ExtremalMask(f).attribute(
        gate, baselines=torch.zeros_like(gate), additional_forward_args=_extra(fixed, space, "ts"),
        trainer=_lightning_trainer(cfg.mask_epochs), mask_net=net, batch_size=min(cfg.mask_batch_size, n),
    )  # fmt: skip
    return _block_mean(space, hourly.detach().numpy())


def ts_dynamask(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    """tint's DynaMask: a mask learned over the *values*, perturbed by temporal smoothing (its
    own perturbation, not our removal), so it is a comparison method only. ``[N, n_ts]``."""
    from tint.attr import DynaMask

    torch.manual_seed(cfg.seed)
    n = fixed["values"].shape[0]
    g = ones_gates(space.layout, n)

    def f(values, ts_gate, static_gate, concept_gate, present, mask, delta, static, image):
        logit = model(ts_gate, static_gate, concept_gate, present, values, mask, delta, static, image)
        return logit.unsqueeze(-1)

    extra = (g["ts"], g["static"], g["concept"], fixed["image_present"], fixed["mask"], fixed["delta"])
    extra += (fixed["static"], fixed["image"])
    hourly = DynaMask(f).attribute(
        fixed["values"],
        additional_forward_args=extra,
        trainer=_lightning_trainer(cfg.dynamask_epochs),
        batch_size=min(cfg.mask_batch_size, n),
    )
    return _block_mean(space, hourly.detach().numpy())


# ---- static groups and image concepts ------------------------------------------------------


def static_ablation(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    return BatchScorer(model, space, fixed).delta(np.arange(space.n_ts, space.n_ts + space.n_static))


def static_ig(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    n = fixed["values"].shape[0]
    return _ig(model, space, fixed, "static", ones_gates(space.layout, n)["static"], cfg.ig_steps)


def concept_ablation(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    return BatchScorer(model, space, fixed).delta(
        np.arange(space.concept_slice.start, space.concept_slice.stop)
    )


def concept_ig(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    n = fixed["values"].shape[0]
    return _ig(model, space, fixed, "concept", ones_gates(space.layout, n)["concept"], cfg.ig_steps)


def whole_image_ablation(model: GatedLogit, space: UnitSpace, fixed: dict, cfg: MethodConfig) -> np.ndarray:
    if not space.has_whole_image:
        raise ValueError("this unit space has no whole-image unit")
    return BatchScorer(model, space, fixed).delta(np.array([space.whole_image_index]))[:, 0]


TS = {"occlusion": ts_occlusion, "ig": ts_ig, "extremal_mask": ts_extremal_mask, "dynamask": ts_dynamask}
STATIC = {"ablation": static_ablation, "ig": static_ig}
CONCEPT = {"ablation": concept_ablation, "ig": concept_ig}


def attribute(
    model: GatedLogit,
    space: UnitSpace,
    fixed: dict[str, torch.Tensor],
    cfg: MethodConfig,
    *,
    ts: str = "extremal_mask",
    static: str = "ig",
    concept: str = "ig",
) -> np.ndarray:
    """One score per unit of the space for each stay: ``[N, n_units]`` (whole image: Δ)."""
    for name, table, choice in (
        ("time series", TS, ts),
        ("static", STATIC, static),
        ("image", CONCEPT, concept),
    ):
        if choice not in table:
            raise ValueError(f"unknown {name} method {choice!r}; choose from {sorted(table)}")
    n = fixed["values"].shape[0]
    out = np.zeros((n, len(space)), dtype=np.float32)
    out[:, space.ts_slice] = TS[ts](model, space, fixed, cfg)
    out[:, space.static_slice] = STATIC[static](model, space, fixed, cfg)
    if "image" in model.net.spec.modalities:  # an EHR-only arm ignores the image: its image units score 0
        out[:, space.concept_slice] = CONCEPT[concept](model, space, fixed, cfg)
        if space.has_whole_image:
            out[:, space.whole_image_index] = whole_image_ablation(model, space, fixed, cfg)
    return out
