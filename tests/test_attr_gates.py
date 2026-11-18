"""Stage 4 gates: a gate of 0 is exactly the removal the model was trained with."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from stay_data import GROUPS, HOURS, IMAGE_DIM, INDICATOR_COLUMNS, random_stay_data

from fate_icu.attribution.gates import GatedLogit, build_layout, logit, ones_gates, stay_tensors
from fate_icu.models.fusion import ModelSpec
from fate_icu.models.removal import remove_static_groups, remove_ts_units
from fate_icu.models.train import TrainConfig, predict, train_model

WIDTH, CONCEPTS = 9, ["c_a", "c_b", "c_c"]  # the synthetic image vector: 9 pooled + 3 concept logits
SMALL = dict(d=16, heads=2, layers=1, dropout=0.0)
FAST = TrainConfig(epochs=3, batch_size=16, patience=3, augment=True)


@pytest.fixture(scope="module")
def data():
    d = random_stay_data(n=100, seed=31, image_share=0.7)
    assert WIDTH + len(CONCEPTS) == IMAGE_DIM
    return d


@pytest.fixture(scope="module")
def trained(data):
    return train_model(ModelSpec(("static", "ts", "image"), "attention", **SMALL), data, FAST, seed=0)


@pytest.fixture(scope="module")
def layout(data):
    return build_layout(data, CONCEPTS, feature_width=WIDTH)


@pytest.fixture(scope="module")
def model(trained, layout):
    return GatedLogit(trained.model, layout)


@pytest.fixture(scope="module")
def rows():
    return np.arange(12)


def test_all_gates_open_reproduces_the_model_logit(model, trained, data, rows):
    fixed = stay_tensors(data, rows)
    with torch.no_grad():
        gated = logit(model, fixed).numpy()
    assert np.allclose(gated, predict(trained, data, rows)["logit"], atol=1e-5)


def test_a_closed_time_series_gate_is_the_training_time_removal(model, data, layout, rows):
    fixed = stay_tensors(data, rows)
    gate = torch.ones(len(rows), HOURS, len(layout.channels))
    gate[:, 4:8, 1] = 0.0
    gate[3, 12:16, :] = 0.0
    values, mask, delta = remove_ts_units(
        fixed["values"], fixed["mask"], fixed["delta"], gate == 0, layout.delta_fill
    )
    reference = {**fixed, "values": values, "mask": mask, "delta": delta}
    with torch.no_grad():
        assert torch.allclose(logit(model, fixed, ts=gate), logit(model, reference), atol=1e-6)


def test_a_closed_static_gate_is_the_training_time_removal(model, data, layout, rows):
    fixed = stay_tensors(data, rows)
    gate = torch.ones(len(rows), len(layout.groups))
    gate[:, 0] = 0.0
    gate[5, 2] = 0.0
    names = list(data.static_groups)
    removed = torch.from_numpy(gate.numpy() == 0)
    static = remove_static_groups(
        fixed["static"], removed, [data.static_groups[g] for g in names],
        torch.from_numpy(data.static_fill), data.indicator_columns,
    )  # fmt: skip
    with torch.no_grad():
        assert torch.allclose(
            logit(model, fixed, static=gate), logit(model, {**fixed, "static": static}), atol=1e-6
        )
    assert names == list(GROUPS)


def test_a_closed_concept_gate_takes_the_train_mean_logit(model, data, layout, rows):
    fixed = stay_tensors(data, rows)
    gate = torch.ones(len(rows), len(CONCEPTS))
    gate[:, 1] = 0.0
    image = fixed["image"].clone()
    image[:, WIDTH + 1] = torch.from_numpy(layout.concept_fill)[1]
    with torch.no_grad():
        assert torch.allclose(
            logit(model, fixed, concept=gate), logit(model, {**fixed, "image": image}), atol=1e-6
        )
    train = data.rows("train", paired_only=True)
    assert np.allclose(layout.concept_fill, data.image[train][:, WIDTH:].mean(axis=0))


def test_removing_the_whole_image_marks_it_absent(model, data, rows):
    fixed = stay_tensors(data, rows)
    absent = torch.zeros(len(rows), dtype=torch.bool)
    scrambled = {**fixed, "image": fixed["image"] * 7 + 1}
    with torch.no_grad():
        assert torch.allclose(
            logit(model, fixed, image_present=absent),
            logit(model, scrambled, image_present=absent),
            atol=1e-6,
        )


def test_gates_are_differentiable_and_gradients_reach_every_gate(model, data, layout, rows):
    fixed = stay_tensors(data, rows)
    gates = {k: v.clone().requires_grad_(True) for k, v in ones_gates(layout, len(rows)).items()}
    out = logit(model, fixed, ts=gates["ts"], static=gates["static"], concept=gates["concept"]).sum()
    out.backward()
    assert all(g.grad is not None and torch.isfinite(g.grad).all() for g in gates.values())
    assert gates["ts"].grad.abs().sum() > 0 and gates["static"].grad.abs().sum() > 0
    assert gates["concept"].grad[torch.from_numpy(data.has_image[rows])].abs().sum() > 0
    assert not any(p.requires_grad for p in model.net.parameters())  # the model itself is frozen


def test_layout_maps_every_static_column_to_one_group(layout, data):
    n_inputs = data.static.shape[1] + data.static_missing.shape[1]
    assert layout.group_of_column.shape == (n_inputs,) and (layout.group_of_column >= 0).all()
    for k, column in enumerate(INDICATOR_COLUMNS):  # an indicator belongs to its column's group
        assert layout.group_of_column[data.static.shape[1] + k] == layout.group_of_column[column]
    assert layout.removed_static[data.static.shape[1] :].tolist() == [1.0] * len(INDICATOR_COLUMNS)
    assert layout.delta_fill == float(data.delta.max())


def test_layout_refuses_a_wrong_image_width_or_no_train_images(data):
    with pytest.raises(ValueError, match="expected"):
        build_layout(data, CONCEPTS[:2], feature_width=WIDTH)
    none = random_stay_data(n=60, seed=2, image_share=0.0)
    with pytest.raises(ValueError, match="no train stay has an image"):
        build_layout(none, CONCEPTS, feature_width=WIDTH)
