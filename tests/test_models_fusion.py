"""Stage 3 encoders and fusion: shapes, missing-modality masking, spec checks."""

from __future__ import annotations

import pytest
import torch
from stay_data import random_stay_data

from fate_icu.models.fusion import FusionNet, ModelSpec
from fate_icu.models.train import _present, _tensors

D = 32


@pytest.fixture(scope="module")
def data():
    return random_stay_data(n=40, seed=3)


def make(spec_kwargs, data):
    torch.manual_seed(0)
    spec = ModelSpec(d=D, heads=4, layers=2, dropout=0.0, **spec_kwargs)
    return FusionNet(spec, data.dims).eval()


def inputs(data, rows=slice(0, 10)):
    rows = list(range(len(data)))[rows]
    import numpy as np

    rows = np.array(rows)
    return _tensors(data, rows, torch.device("cpu")), _present(data, rows, torch.device("cpu"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"modalities": ("static",), "fusion": "none"},
        {"modalities": ("ts",), "fusion": "none"},
        {"modalities": ("ts",), "fusion": "none", "ts_encoder": "transformer"},
        {"modalities": ("image",), "fusion": "none"},
        {"modalities": ("static", "ts"), "fusion": "concat"},
        {"modalities": ("static", "ts", "image"), "fusion": "concat"},
        {"modalities": ("static", "ts", "image"), "fusion": "attention"},
        {"modalities": ("static", "ts", "image"), "fusion": "attention", "ts_encoder": "transformer"},
    ],
)
def test_every_model_gives_one_logit_and_one_log_los_per_stay(kwargs, data):
    net = make(kwargs, data)
    batch, present = inputs(data)
    with torch.no_grad():
        out = net(batch, present)
    assert out["logit"].shape == (10,) and out["log_los"].shape == (10,)
    assert torch.isfinite(out["logit"]).all() and torch.isfinite(out["log_los"]).all()


def test_the_lstm_token_has_the_requested_width(data):
    net = make({"modalities": ("ts",), "fusion": "none"}, data)
    batch, _ = inputs(data)
    assert net.tokens(batch)["ts"].shape == (10, D)


@pytest.mark.parametrize("fusion", ["attention", "concat"])
def test_a_removed_image_cannot_influence_the_output(fusion, data):
    net = make({"modalities": ("static", "ts", "image"), "fusion": fusion}, data)
    batch, present = inputs(data)
    present = {**present, "image": torch.zeros(10, dtype=torch.bool)}
    scrambled = {**batch, "image": torch.randn_like(batch["image"]) * 50}
    with torch.no_grad():
        a, b = net(batch, present), net(scrambled, present)
    assert torch.allclose(a["logit"], b["logit"], atol=1e-6)
    assert torch.allclose(a["log_los"], b["log_los"], atol=1e-6)


def test_attention_with_a_missing_image_equals_a_model_never_given_one(data):
    # the masked token must be removed, not just down-weighted: with the image absent the
    # output must not move when the image encoder's weights change
    net = make({"modalities": ("static", "ts", "image"), "fusion": "attention"}, data)
    batch, present = inputs(data)
    present = {**present, "image": torch.zeros(10, dtype=torch.bool)}
    with torch.no_grad():
        before = net(batch, present)["logit"].clone()
        for parameter in net.encoders["image"].parameters():
            parameter.add_(1.0)
        after = net(batch, present)["logit"]
    assert torch.allclose(before, after, atol=1e-6)


def test_other_modalities_still_matter_when_the_image_is_removed(data):
    net = make({"modalities": ("static", "ts", "image"), "fusion": "attention"}, data)
    batch, present = inputs(data)
    present = {**present, "image": torch.zeros(10, dtype=torch.bool)}
    changed = {**batch, "static": batch["static"] + 3.0}
    with torch.no_grad():
        assert not torch.allclose(net(batch, present)["logit"], net(changed, present)["logit"], atol=1e-4)


def test_a_present_image_matters(data):
    net = make({"modalities": ("static", "ts", "image"), "fusion": "attention"}, data)
    batch, present = inputs(data)
    present = {**present, "image": torch.ones(10, dtype=torch.bool)}
    changed = {
        **batch,
        "image": torch.randn_like(batch["image"]) * 3,
    }  # not a constant shift: LayerNorm removes that
    with torch.no_grad():
        assert not torch.allclose(net(batch, present)["logit"], net(changed, present)["logit"], atol=1e-4)


def test_stays_do_not_influence_each_other_in_a_batch(data):
    net = make({"modalities": ("static", "ts", "image"), "fusion": "attention"}, data)
    batch, present = inputs(data)
    with torch.no_grad():
        together = net(batch, present)["logit"]
        alone = net({k: v[:1] for k, v in batch.items()}, {k: v[:1] for k, v in present.items()})["logit"]
    assert torch.allclose(together[:1], alone, atol=1e-5)


def test_spec_validation():
    with pytest.raises(ValueError, match="non-empty subset"):
        ModelSpec(modalities=("audio",), fusion="none")
    with pytest.raises(ValueError, match="exactly one"):
        ModelSpec(modalities=("static", "ts"), fusion="none")
    with pytest.raises(ValueError, match="exactly one"):
        ModelSpec(modalities=("static",), fusion="concat")
    with pytest.raises(ValueError, match="unknown fusion"):
        ModelSpec(modalities=("static",), fusion="sum")
    with pytest.raises(ValueError, match="divisible"):
        ModelSpec(d=30, heads=4)
    with pytest.raises(ValueError, match="time-series encoder"):
        ModelSpec(modalities=("ts",), fusion="none", ts_encoder="gru")
