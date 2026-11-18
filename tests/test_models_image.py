"""Stage 3 image encoder: frozen DenseNet-121 (chex), feature shape and the feature cache.

Needs the ``chex`` weights (28 MB). They download on first use; a machine that is offline and
has not downloaded them yet skips these tests, naming the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fate_icu.data.cxr.cache import fixture_items
from fate_icu.data.cxr.preprocess import build_cache, load_cached
from fate_icu.models.image import FEATURE_WIDTH, WEIGHTS, cached_features, feature_path, load_image_encoder
from fate_icu.utils.env import REPO_ROOT

WEIGHT_FILE = (
    Path.home()
    / ".torchxrayvision"
    / "models_data"
    / ("chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt")
)


@pytest.fixture(scope="module")
def encoder():
    if not WEIGHT_FILE.exists():
        pytest.skip(f"download the chex weights first (python -m fate_icu.models.image): {WEIGHT_FILE}")
    return load_image_encoder()


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cxr_cache")
    items = fixture_items(REPO_ROOT / "tests" / "fixtures" / "cxr")
    summary = build_cache(items, folder)
    assert not summary.missing and not summary.failed
    return folder, [key for key, _ in items]


def test_the_feature_vector_is_pooled_features_plus_the_non_empty_pathology_logits(encoder):
    assert WEIGHTS == "densenet121-res224-chex" == encoder.weights
    assert len(encoder.pathology_index) == len(encoder.pathology_names) == 11
    assert "Edema" in encoder.pathology_names and "" not in encoder.pathology_names
    assert encoder.dim == FEATURE_WIDTH + 11
    out = encoder.encode(np.zeros((2, 1, 224, 224), dtype=np.float32))
    assert out.shape == (2, encoder.dim) and out.dtype == np.float32 and np.isfinite(out).all()


def test_the_encoder_is_frozen_and_deterministic(encoder, cache):
    assert not any(p.requires_grad for p in encoder.model.parameters()) and not encoder.model.training
    folder, keys = cache
    x = np.stack([load_cached(folder, keys[0])["x224"]])
    assert np.array_equal(encoder.encode(x), encoder.encode(x))


def test_cached_features_match_on_the_fly_features(encoder, cache, tmp_path):
    folder, keys = cache
    computed = cached_features(keys, folder, tmp_path, encoder)  # computes and stores
    again = cached_features(keys, folder, tmp_path, encoder)  # read back from disk
    for key in keys:
        on_the_fly = encoder.encode(load_cached(folder, key)["x224"][None], batch_size=1)[0]
        assert np.array_equal(computed[key], again[key])
        assert np.allclose(computed[key], on_the_fly, atol=1e-4)
        assert feature_path(tmp_path, encoder.weights, key).exists()


def test_the_pathology_logits_are_the_raw_classifier_outputs(encoder, cache):
    folder, keys = cache
    x = load_cached(folder, keys[1])["x224"][None]
    import torch

    with torch.no_grad():
        pooled = encoder.model.features2(torch.from_numpy(x))
        raw = encoder.model.classifier(pooled)[:, encoder.pathology_index].numpy()
    vector = encoder.encode(x)[0]
    assert np.allclose(vector[FEATURE_WIDTH:], raw[0], atol=1e-5)
    assert np.allclose(vector[:FEATURE_WIDTH], pooled[0].numpy(), atol=1e-5)


def test_features_differ_between_images(encoder, cache, tmp_path):
    folder, keys = cache
    features = cached_features(keys, folder, tmp_path, encoder)
    stacked = np.stack([features[k] for k in keys])
    assert len(keys) >= 2 and np.unique(stacked.round(4), axis=0).shape[0] == len(keys)


def test_a_missing_image_stops_the_run(encoder, cache, tmp_path):
    folder, _ = cache
    with pytest.raises(FileNotFoundError):
        cached_features(["no_such_image"], folder, tmp_path, encoder)


def test_the_cache_is_separated_by_weights_name(tmp_path):
    assert feature_path(tmp_path, "a", "abcd") != feature_path(tmp_path, "b", "abcd")
    assert feature_path(tmp_path, "a", "abcd").parent.name == "ab"


def test_the_wrong_image_shape_is_refused(encoder):
    with pytest.raises(ValueError, match="224"):
        encoder.encode(np.zeros((1, 1, 100, 100), dtype=np.float32))
