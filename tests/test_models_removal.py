"""What "removed" means as a model input, and the training-time dropout that uses it."""

from __future__ import annotations

import numpy as np
import torch
from stay_data import GROUPS, INDICATOR_COLUMNS, N_STATIC, random_stay_data

from fate_icu.models.fusion import ModelSpec
from fate_icu.models.removal import block_unit_mask, remove_static_groups, remove_ts_units
from fate_icu.models.train import TrainConfig, _present, _tensors, augment


def test_block_units_cover_whole_four_hour_blocks():
    generator = torch.Generator().manual_seed(0)
    units = block_unit_mask(50, 24, 7, 4, torch.full((50,), 0.5), generator)
    assert units.shape == (50, 24, 7)
    blocks = units.reshape(50, 6, 4, 7)
    assert (blocks == blocks[:, :, :1, :]).all()  # all four hours of a block agree


def test_a_last_short_block_is_cut_to_the_grid():
    units = block_unit_mask(4, 10, 3, 4, torch.full((4,), 0.5), torch.Generator().manual_seed(1))
    assert units.shape == (4, 10, 3)


def test_removal_fraction_follows_the_requested_probability():
    generator = torch.Generator().manual_seed(2)
    units = block_unit_mask(4000, 24, 8, 4, torch.full((4000,), 0.125), generator)
    assert abs(units.float().mean().item() - 0.125) < 0.01


def test_removing_ts_units_looks_like_a_never_observed_channel():
    values, mask, delta = torch.ones(2, 8, 3), torch.ones(2, 8, 3), torch.zeros(2, 8, 3)
    units = torch.zeros(2, 8, 3, dtype=torch.bool)
    units[0, 2:6, 1] = True
    v, m, d = remove_ts_units(values, mask, delta, units, delta_fill=1.0)
    assert (v[0, 2:6, 1] == 0).all() and (m[0, 2:6, 1] == 0).all() and (d[0, 2:6, 1] == 1).all()
    assert v.sum() == values.sum() - 4 and m.sum() == mask.sum() - 4
    assert (values == 1).all()  # the inputs are not changed in place
    untouched = ~units
    assert (v[untouched] == 1).all() and (d[untouched] == 0).all()


def test_removing_a_static_group_fills_values_and_flags_the_indicators():
    static = torch.cat([torch.ones(3, N_STATIC) * 5, torch.zeros(3, len(INDICATOR_COLUMNS))], dim=1)
    fill = torch.arange(N_STATIC, dtype=torch.float32)
    removed = torch.tensor([[True, False, False], [False, False, False], [False, True, False]])
    out = remove_static_groups(static, removed, list(GROUPS.values()), fill, INDICATOR_COLUMNS)
    # row 0 loses demographics (columns 0, 1); column 1 has an indicator (position 0)
    assert out[0, 0] == 0 and out[0, 1] == 1 and out[0, N_STATIC + 0] == 1 and out[0, N_STATIC + 1] == 0
    assert torch.equal(out[1], static[1])
    # row 2 loses vitals (columns 2, 3): neither has an indicator
    assert out[2, 2] == 2 and out[2, 3] == 3 and out[2, N_STATIC:].sum() == 0
    assert (static[:, :N_STATIC] == 5).all()  # the input is not changed in place


def _augmented(spec, seed, cfg=None, n=200):
    data = random_stay_data(n=n, seed=1)
    cfg = cfg or TrainConfig()
    rows = np.arange(n)
    batch, present = _tensors(data, rows, torch.device("cpu")), _present(data, rows, torch.device("cpu"))
    return (
        data,
        batch,
        present,
        *augment(batch, present, spec, data, cfg, torch.Generator().manual_seed(seed)),
    )


def test_modality_dropout_never_removes_everything_and_never_adds_an_image():
    spec = ModelSpec(("static", "ts", "image"), "attention", d=16)
    cfg = TrainConfig(p_drop_static=0.9, p_drop_ts=0.9, p_drop_image=0.9)
    data, _, present, _, new_present = _augmented(spec, 0, cfg)
    stacked = torch.stack([new_present[m] for m in spec.modalities], dim=1)
    assert stacked.any(dim=1).all()  # at least one modality survives in every stay
    assert not (new_present["image"] & ~present["image"]).any()  # a missing image stays missing


def test_modality_dropout_rates_are_close_to_the_configured_ones():
    spec = ModelSpec(("static", "ts", "image"), "attention", d=16)
    data = random_stay_data(n=4000, seed=2, image_share=1.0)
    cfg = TrainConfig()
    rows = np.arange(len(data))
    batch, present = _tensors(data, rows, torch.device("cpu")), _present(data, rows, torch.device("cpu"))
    _, new = augment(batch, present, spec, data, cfg, torch.Generator().manual_seed(4))
    assert abs(1 - new["image"].float().mean().item() - 0.3) < 0.03
    assert abs(1 - new["ts"].float().mean().item() - 0.1) < 0.03
    assert abs(1 - new["static"].float().mean().item() - 0.1) < 0.03


def test_single_modality_models_get_no_modality_dropout():
    spec = ModelSpec(("static",), "none", d=16)
    _, _, _, _, new_present = _augmented(spec, 0, TrainConfig(p_drop_static=1.0))
    assert new_present["static"].all()


def test_unit_dropout_removes_about_ten_to_fifteen_percent_of_observed_units():
    spec = ModelSpec(("ts",), "none", d=16)
    data, batch, _, new_batch, _ = _augmented(spec, 5, n=300)
    removed = (new_batch["mask"] == 0) & (batch["mask"] == 1)
    observed = batch["mask"].sum().item()
    assert 0.09 < removed.sum().item() / observed < 0.16
    # removed cells are zeroed everywhere they were removed, and nothing else changes
    changed = new_batch["values"] != batch["values"]
    assert (new_batch["values"][changed] == 0).all()


def test_augmentation_is_deterministic_for_a_seed_and_differs_across_seeds():
    spec = ModelSpec(("static", "ts", "image"), "attention", d=16)
    _, _, _, a, _ = _augmented(spec, 7)
    _, _, _, b, _ = _augmented(spec, 7)
    _, _, _, c, _ = _augmented(spec, 8)
    assert torch.equal(a["values"], b["values"]) and not torch.equal(a["values"], c["values"])
