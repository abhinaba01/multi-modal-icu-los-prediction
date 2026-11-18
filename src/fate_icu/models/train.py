"""Training and prediction for one model on one seed (Stage 3).

Loss = BCE (``pos_weight`` = train negatives / train positives) + lambda * Huber on
``log1p(LOS days)``. AdamW, linear warmup then cosine decay, early stopping on validation
AUPRC (patience 8). Nothing is tuned on test; the test split is never read here.

Robustness training (required for valid deletion tests, docs/stages.md):

* modality dropout: image p = 0.3, static and time series p = 0.1, never all at once;
* evidence-unit dropout: each step, each stay loses a random 10-15 % of its (variable x
  4-hour block) units and each static group with probability ``static_group_drop``.

What "removed" means as an input is defined once, in ``fate_icu.models.removal``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from fate_icu.models.data import StayData
from fate_icu.models.fusion import FusionNet, ModelSpec
from fate_icu.models.removal import block_unit_mask, remove_static_groups, remove_ts_units
from fate_icu.utils.device import get_device
from fate_icu.utils.seed import seed_everything

CHECKPOINT_FORMAT = 1
LOS_CAP_LOG = 8.0  # expm1 guard: exp(8) - 1 is about 2980 days


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    patience: int = 8
    lambda_reg: float = 1.0
    huber_delta: float = 1.0
    p_drop_static: float = 0.1
    p_drop_ts: float = 0.1
    p_drop_image: float = 0.3
    unit_drop_low: float = 0.10
    unit_drop_high: float = 0.15
    block_hours: int = 4
    static_group_drop: float = 0.1
    augment: bool = True
    selection: str = "auprc"  # the validation metric that picks the best epoch
    allow_loss_fallback: bool = False  # smoke only: validation loss when val has one class
    autocast_bf16: bool = False  # bf16 autocast on CUDA (the lab config); ignored on CPU
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")
        if not 0 <= self.unit_drop_low <= self.unit_drop_high < 1:
            raise ValueError("need 0 <= unit_drop_low <= unit_drop_high < 1")
        if self.selection != "auprc":
            raise ValueError("only validation AUPRC is implemented as the selection metric")


def _tensors(data: StayData, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    static = np.concatenate([data.static[rows], data.static_missing[rows]], axis=1)
    return {
        "static": torch.from_numpy(static).to(device),
        "values": torch.from_numpy(data.values[rows]).to(device),
        "mask": torch.from_numpy(data.mask[rows]).to(device),
        "delta": torch.from_numpy(data.delta[rows]).to(device),
        "image": torch.from_numpy(data.image[rows]).to(device),
    }


def _present(data: StayData, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    n = len(rows)
    return {
        "static": torch.ones(n, dtype=torch.bool, device=device),
        "ts": torch.ones(n, dtype=torch.bool, device=device),
        "image": torch.from_numpy(data.has_image[rows]).to(device),
    }


def augment(
    batch: dict[str, torch.Tensor],
    present: dict[str, torch.Tensor],
    spec: ModelSpec,
    data: StayData,
    cfg: TrainConfig,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Modality dropout and evidence-unit dropout for one training batch."""
    n = batch["static"].shape[0]
    present = dict(present)
    batch = dict(batch)
    if len(spec.modalities) > 1:
        probs = {"static": cfg.p_drop_static, "ts": cfg.p_drop_ts, "image": cfg.p_drop_image}
        drop = {
            m: (torch.rand(n, generator=generator) < probs[m]).to(batch["static"].device)
            for m in spec.modalities
        }
        kept = torch.stack([present[m] & ~drop[m] for m in spec.modalities], dim=1)
        # never all at once: a stay that lost everything keeps a random modality it has
        lost = ~kept.any(dim=1)
        if lost.any():
            have = torch.stack([present[m] for m in spec.modalities], dim=1).float()
            noise = torch.rand(have.shape, generator=generator).to(have.device)
            pick = torch.argmax(have * noise, dim=1)
            for k, m in enumerate(spec.modalities):
                drop[m] = drop[m] & ~(lost & (pick == k))
        for m in spec.modalities:
            present[m] = present[m] & ~drop[m]
    hours, channels = batch["values"].shape[1:]
    if "ts" in spec.modalities:
        fraction = cfg.unit_drop_low + (cfg.unit_drop_high - cfg.unit_drop_low) * torch.rand(
            n, generator=generator
        )
        units = block_unit_mask(n, hours, channels, cfg.block_hours, fraction, generator).to(
            batch["values"].device
        )
        delta_fill = float(data.delta.max()) if data.delta.size else 1.0
        batch["values"], batch["mask"], batch["delta"] = remove_ts_units(
            batch["values"], batch["mask"], batch["delta"], units, delta_fill
        )
    if "static" in spec.modalities and data.static_groups and data.static_fill is not None:
        names = list(data.static_groups)
        removed = (torch.rand(n, len(names), generator=generator) < cfg.static_group_drop).to(
            batch["static"].device
        )
        batch["static"] = remove_static_groups(
            batch["static"],
            removed,
            [data.static_groups[g] for g in names],
            torch.from_numpy(data.static_fill).to(batch["static"].device),
            data.indicator_columns,
        )
    return batch, present


@dataclass
class TrainedModel:
    """A fitted network with its training history."""

    model: FusionNet
    spec: ModelSpec
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    selection: str = "auprc"
    device: str = "cpu"


@torch.no_grad()
def predict(
    trained: TrainedModel, data: StayData, rows: np.ndarray, batch_size: int = 256
) -> dict[str, np.ndarray]:
    """``p_long``, ``logit`` and ``los_days`` for ``rows``; absent modalities are left out.

    A model that needs a modality a stay lacks (e.g. the image-only arm) gives NaN there.
    """
    model, device = trained.model.eval(), torch.device(trained.device)
    logits, log_los = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        out = model(_tensors(data, chunk, device), _present(data, chunk, device))
        logits.append(out["logit"].cpu().numpy())
        log_los.append(out["log_los"].cpu().numpy())
    logit = np.concatenate(logits) if logits else np.zeros(0, np.float32)
    days = np.expm1(np.clip(np.concatenate(log_los) if log_los else np.zeros(0), 0.0, LOS_CAP_LOG))
    need = [
        data.has_image[rows] if m == "image" else np.ones(len(rows), bool) for m in trained.spec.modalities
    ]
    usable = np.logical_and.reduce(need) if need else np.ones(len(rows), bool)
    if trained.spec.fusion == "none":
        logit, days = np.where(usable, logit, np.nan), np.where(usable, days, np.nan)
    return {
        "logit": logit.astype(np.float64),
        "p_long": 1 / (1 + np.exp(-logit.astype(np.float64))),
        "los_days": days.astype(np.float64),
    }


def _validation_score(
    trained_model: FusionNet,
    data: StayData,
    rows: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
    loss_fn,
) -> tuple[float, float]:
    """``(auprc or nan, validation loss)``."""
    trained = TrainedModel(trained_model, trained_model.spec, device=str(device))
    out = predict(trained, data, rows)
    y = data.y_long[rows]
    ok = ~np.isnan(out["p_long"])
    auprc = (
        float(average_precision_score(y[ok], out["p_long"][ok]))
        if len(np.unique(y[ok])) > 1
        else float("nan")
    )
    target = np.log1p(data.los_days[rows])
    loss = float(np.mean(np.abs(np.log1p(out["los_days"][ok]) - target[ok]))) if ok.any() else float("nan")
    return auprc, loss


def train_model(
    spec: ModelSpec,
    data: StayData,
    cfg: TrainConfig,
    seed: int,
    *,
    train_rows: np.ndarray | None = None,
    val_rows: np.ndarray | None = None,
) -> TrainedModel:
    """Fit ``spec`` on the train stays (or ``train_rows``) and keep the best validation epoch."""
    seed_everything(seed)
    device = torch.device(get_device(cfg.device))
    train_rows = data.rows("train") if train_rows is None else np.asarray(train_rows)
    val_rows = data.rows("val") if val_rows is None else np.asarray(val_rows)
    if len(train_rows) < 2:
        raise ValueError("need at least two train stays")
    y_train = data.y_long[train_rows]
    n_pos = float(y_train.sum())
    if n_pos == 0 or n_pos == len(y_train):
        raise ValueError("the train stays need both short and long stays")
    pos_weight = torch.tensor((len(y_train) - n_pos) / n_pos, device=device)
    generator = torch.Generator().manual_seed(seed)
    use_autocast = cfg.autocast_bf16 and device.type == "cuda"

    model = FusionNet(spec, data.dims).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_factor)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber = nn.HuberLoss(delta=cfg.huber_delta)
    log_los_all = torch.from_numpy(np.log1p(data.los_days)).float()
    y_all = torch.from_numpy(data.y_long).float()

    history: list[dict[str, float]] = []
    best_state, best_epoch, best_score, stale = None, -1, -float("inf"), 0
    selection = "auprc"
    for epoch in range(cfg.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            if len(rows) < 2:
                continue  # a batch of one has no variance for the normalisation layers
            batch, present = _tensors(data, rows, device), _present(data, rows, device)
            if cfg.augment:
                batch, present = augment(batch, present, spec, data, cfg, generator)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                out = model(batch, present)
            out = {k: v.float() for k, v in out.items()}
            loss = bce(out["logit"], y_all[rows].to(device)) + cfg.lambda_reg * huber(
                out["log_los"], log_los_all[rows].to(device)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            running += loss.item() * len(rows)
        auprc, val_mae_log = _validation_score(model, data, val_rows, device, cfg, None)
        if math.isnan(auprc):
            if not cfg.allow_loss_fallback:
                raise ValueError(
                    "the validation stays have one class only, so AUPRC is undefined; "
                    "set allow_loss_fallback only for smoke runs"
                )
            selection, score = "val_mae_log1p", -val_mae_log
        else:
            score = auprc
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(order),
                "val_auprc": auprc,
                "val_mae_log1p": val_mae_log,
            }
        )
        if score > best_score + 1e-9:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainedModel(model.eval(), spec, history, best_epoch, selection, str(device))


def save_trained(
    trained: TrainedModel, path: str | Path, dims: dict[str, int], extra: dict | None = None
) -> Path:
    """Write a trained network to ``path`` (weights, spec, input sizes and history).

    The file holds model weights derived from credentialed data in a real run, so it belongs
    in the git-ignored checkpoints folder (CLAUDE.md A3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state": {k: v.detach().cpu() for k, v in trained.model.state_dict().items()},
        "spec": asdict(trained.spec),
        "dims": dict(dims),
        "history": trained.history,
        "best_epoch": trained.best_epoch,
        "selection": trained.selection,
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_trained(path: str | Path, device: str = "cpu") -> tuple[TrainedModel, dict]:
    """Read a network written by :func:`save_trained`; returns it with its ``extra`` record."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {payload.get('format')!r}")
    spec_fields = dict(payload["spec"])
    spec_fields["modalities"] = tuple(spec_fields["modalities"])
    spec = ModelSpec(**spec_fields)
    model = FusionNet(spec, payload["dims"]).to(device)
    model.load_state_dict(payload["state"])
    trained = TrainedModel(
        model.eval(), spec, payload["history"], payload["best_epoch"], payload["selection"], device
    )
    return trained, payload["extra"]
