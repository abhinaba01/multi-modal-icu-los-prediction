"""Stage 4 image attribution: Grad-CAM through the real DenseNet, regions, and the finding rule.

The network tests need the ``chex`` weights (and PSPNet for the region tests); they skip, naming
the file, on a machine that has not downloaded them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from stay_data import random_stay_data

from fate_icu.attribution import image_cam, regions
from fate_icu.attribution.gates import GatedLogit, build_layout, stay_tensors
from fate_icu.data.cxr.cache import fixture_items
from fate_icu.data.cxr.preprocess import build_cache, load_cached
from fate_icu.models.fusion import FusionNet, ModelSpec
from fate_icu.models.image import FEATURE_WIDTH, load_image_encoder
from fate_icu.utils.env import REPO_ROOT

MODELS = Path.home() / ".torchxrayvision" / "models_data"
CHEX = MODELS / "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
PSP = MODELS / "pspnet_chestxray_best_model_4.pth"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cxr")
    items = fixture_items(REPO_ROOT / "tests" / "fixtures" / "cxr")
    build_cache(items, folder)
    return folder, [k for k, _ in items]


@pytest.fixture(scope="module")
def encoder():
    if not CHEX.exists():
        pytest.skip(f"download the chex weights first (python -m fate_icu.models.image): {CHEX}")
    return load_image_encoder()


@pytest.fixture(scope="module")
def stay(encoder):
    torch.manual_seed(0)
    data = random_stay_data(n=40, seed=3, image_share=1.0)
    data = dataclasses.replace(
        data, image=np.random.default_rng(0).normal(size=(40, encoder.dim)).astype(np.float32)
    )
    layout = build_layout(data, encoder.pathology_names, feature_width=FEATURE_WIDTH)
    net = FusionNet(
        ModelSpec(("static", "ts", "image"), "attention", d=16, heads=2, layers=1, dropout=0.0), data.dims
    )
    gated = GatedLogit(net.eval(), layout)
    fixed = stay_tensors(data, np.array([0]))
    return data, gated, fixed


def test_pixels_to_logit_equals_the_logit_from_the_encoded_vector(encoder, cache, stay):
    folder, keys = cache
    data, gated, fixed = stay
    x = np.stack([load_cached(folder, keys[0])["x224"]])
    wrapper = image_cam.ImageToLogit(encoder, gated, fixed)
    with torch.no_grad():
        from_pixels = wrapper(torch.from_numpy(x)).numpy()
        vector = torch.from_numpy(encoder.encode(x))
        from_vector = gated(
            torch.ones(1, 24, 5), torch.ones(1, 3), torch.ones(1, encoder.dim - FEATURE_WIDTH),
            torch.ones(1, dtype=torch.bool),
            fixed["values"], fixed["mask"], fixed["delta"], fixed["static"], vector,
        ).numpy()  # fmt: skip
    assert np.allclose(from_pixels, from_vector, atol=1e-4)


def test_grad_cam_gives_a_normalised_deterministic_heatmap_per_direction(encoder, cache, stay):
    folder, keys = cache
    _, gated, fixed = stay
    wrapper = image_cam.ImageToLogit(encoder, gated, fixed)
    x = load_cached(folder, keys[0])["x224"]
    up = image_cam.grad_cam(wrapper, x, +1)
    again = image_cam.grad_cam(wrapper, x, +1)
    down = image_cam.grad_cam(wrapper, x, -1)
    assert up.shape == (224, 224) and up.dtype == np.float32
    assert up.min() >= 0.0 and up.max() <= 1.0 + 1e-6 and up.std() > 0
    assert np.array_equal(up, again) and not np.allclose(up, down)
    with pytest.raises(ValueError, match="direction"):
        image_cam.grad_cam(wrapper, x, 0)


def test_covering_the_hottest_pixels_moves_the_logit_more_than_the_coldest(encoder, cache, stay):
    folder, keys = cache
    _, gated, fixed = stay
    wrapper = image_cam.ImageToLogit(encoder, gated, fixed)
    wins = 0
    for key in keys:
        x = load_cached(folder, key)["x224"]
        heat = image_cam.grad_cam(wrapper, x, +1)
        order = np.argsort(heat.ravel())
        n = int(0.1 * heat.size)

        def cover(pixels, x=x):
            y = x.copy()
            y.reshape(-1)[pixels] = 0.0  # the mid-grey of the normalised scale
            with torch.no_grad():
                return float(wrapper(torch.from_numpy(y)[None]))

        with torch.no_grad():
            base = float(wrapper(torch.from_numpy(x)[None]))
        wins += abs(cover(order[-n:]) - base) >= abs(cover(order[:n]) - base)
    assert wins >= len(keys) - 1


def test_operating_thresholds_and_the_finding_rule(encoder):
    thresholds = image_cam.operating_thresholds(encoder)
    assert list(thresholds) == encoder.pathology_names
    assert sum(np.isfinite(v) for v in thresholds.values()) >= 8 and all(
        0 < v < 1 for v in thresholds.values() if np.isfinite(v)
    )
    assert image_cam.finding_allowed(0.6, 0.5) and not image_cam.finding_allowed(0.5, 0.5)
    assert not image_cam.finding_allowed(0.4, 0.5) and not image_cam.finding_allowed(0.99, float("nan"))


def test_concept_probabilities_are_the_sigmoid_of_the_raw_logits():
    vector = np.concatenate([np.zeros(FEATURE_WIDTH), [0.0, 2.0, -2.0]])
    p = image_cam.concept_probabilities(vector, ["a", "b", "c"])
    assert p["a"] == pytest.approx(0.5) and p["b"] == pytest.approx(1 / (1 + np.exp(-2.0))) and p["c"] < 0.5


# ---- regions: the geometry needs no network ---------------------------------------------------


def fake_masks(size=regions.SEGMENTATION_SIZE):
    m = {name: np.zeros((size, size), dtype=bool) for name in (
        "Left Lung", "Right Lung", "Heart", "Mediastinum")}  # fmt: skip
    m["Right Lung"][100:300, 40:200] = True  # the patient's right lung: on the image's left
    m["Left Lung"][100:300, 300:460] = True
    m["Heart"][200:320, 230:330] = True
    m["Mediastinum"][80:320, 215:245] = True
    return m


def test_each_lung_is_split_at_the_middle_of_its_own_extent():
    zones = regions.lung_zones(fake_masks())
    assert set(zones) == {
        "right upper lung zone",
        "right lower lung zone",
        "left upper lung zone",
        "left lower lung zone",
    }
    right_up, right_low = zones["right upper lung zone"], zones["right lower lung zone"]
    assert (
        not (right_up & right_low).any() and (right_up | right_low).sum() == fake_masks()["Right Lung"].sum()
    )
    assert np.flatnonzero(right_up.any(axis=1)).max() < np.flatnonzero(right_low.any(axis=1)).min()
    assert (zones["left upper lung zone"] | zones["left lower lung zone"]).sum() == fake_masks()[
        "Left Lung"
    ].sum()


def test_sides_come_from_the_label_not_from_the_image_columns():
    masks = fake_masks()
    heat = np.zeros((512, 512))
    heat[250:290, 60:180] = 1.0  # inside the lung that is on the image's LEFT
    shares = regions.region_shares(heat, masks)
    assert regions.describe_region(shares) == "right lower lung zone"  # the patient's right
    flipped = {**masks, "Right Lung": masks["Left Lung"], "Left Lung": masks["Right Lung"]}
    assert regions.describe_region(regions.region_shares(heat, flipped)) != "right lower lung zone"


def test_shares_sum_to_one_and_a_scattered_map_has_no_single_region():
    masks = fake_masks()
    rng = np.random.default_rng(0)
    heat = rng.random((224, 224))  # a 224 map is resized to the 512 grid
    shares = regions.region_shares(heat, masks)
    assert sum(shares.values()) == pytest.approx(1.0) and set(shares) >= {
        regions.CARDIOMEDIASTINAL,
        regions.OTHER,
    }
    assert shares[regions.OTHER] > 0.4  # most of a scattered map falls outside the lungs and heart ...
    assert regions.describe_region(shares) == regions.DIFFUSE  # ... which is never named
    outside = np.zeros((512, 512))
    outside[350:480, 20:100] = 1.0  # all of it below the lungs
    assert regions.describe_region(regions.region_shares(outside, masks)) == regions.DIFFUSE
    centre = np.zeros((512, 512))
    centre[240:300, 250:310] = 1.0
    assert regions.describe_region(regions.region_shares(centre, masks)) == regions.CARDIOMEDIASTINAL
    assert (
        regions.region_shares(np.zeros((512, 512)), masks) == {}
        and regions.describe_region({}) == regions.DIFFUSE
    )


def test_a_missing_lung_gives_empty_zones_not_a_crash():
    masks = fake_masks()
    masks["Left Lung"][:] = False
    zones = regions.lung_zones(masks)
    assert not zones["left upper lung zone"].any() and not zones["left lower lung zone"].any()


# ---- regions: the real segmentation network on public films ---------------------------------


def test_the_segmentation_network_labels_are_anatomical_on_public_frontal_films(cache):
    if not PSP.exists():
        pytest.skip(f"download the PSPNet weights first: {PSP}")
    folder, keys = cache
    segmenter = regions.load_segmenter()
    for key in keys:
        masks = segmenter.masks(load_cached(folder, key)["display512"])
        centroid = {n: np.nonzero(masks[n])[1].mean() for n in ("Right Lung", "Left Lung", "Heart")}
        # on a frontal film the patient's right lung is on the image's left, and the heart sits
        # to the image's right of the midline of the two lungs (towards the patient's left)
        assert centroid["Right Lung"] < centroid["Left Lung"], key
        assert centroid["Right Lung"] < centroid["Heart"], key
        assert masks["Right Lung"].sum() > 5000 and masks["Left Lung"].sum() > 5000, key


def test_the_segmenter_checks_its_input_size():
    if not PSP.exists():
        pytest.skip(f"download the PSPNet weights first: {PSP}")
    with pytest.raises(ValueError, match="512"):
        regions.load_segmenter().masks(np.zeros((100, 100), dtype=np.uint8))
