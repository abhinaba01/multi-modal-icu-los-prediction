"""Chest X-ray preprocessing (torchxrayvision conventions) and the image cache."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from fate_icu.data.cxr.cache import fixture_items
from fate_icu.data.cxr.preprocess import (
    _xrv,
    build_cache,
    load_cached,
    preprocess_xray,
    read_xray,
)
from fate_icu.utils.env import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "cxr"


def _normalised(value: float, maxval: float = 255.0) -> float:
    return (2.0 * value / maxval - 1.0) * 1024.0


def test_normalisation_shapes_and_types():
    for value in (0, 128, 255):
        out = preprocess_xray(np.full((300, 200), value, dtype=np.uint8), 255)
        assert out["x224"].shape == (1, 224, 224) and out["x224"].dtype == np.float32
        assert out["display512"].shape == (512, 512) and out["display512"].dtype == np.uint8
        np.testing.assert_allclose(out["x224"], _normalised(value), atol=1e-3)
    with pytest.raises(ValueError):
        preprocess_xray(np.full((10, 10), 300.0), 255)  # above the stated maximum
    with pytest.raises(ValueError):
        preprocess_xray(np.zeros((2, 10, 10)), 255)  # not 2-D


def test_image_modes(tmp_path):
    gray = (np.arange(64 * 48) % 256).astype(np.uint8).reshape(64, 48)
    Image.fromarray(gray, mode="L").save(tmp_path / "l.png")
    array, maxval = read_xray(tmp_path / "l.png")
    assert maxval == 255 and np.array_equal(array, gray)

    deep = (np.arange(64 * 48) * 20).astype(np.uint16).reshape(64, 48)
    Image.fromarray(deep).save(tmp_path / "i16.png")
    array, maxval = read_xray(tmp_path / "i16.png")
    assert maxval == 65535 and np.array_equal(array, deep)

    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 60
    Image.fromarray(rgb, mode="RGB").save(tmp_path / "rgb.png")
    array, maxval = read_xray(tmp_path / "rgb.png")
    assert maxval == 255 and np.allclose(array, 30.0)  # channel mean, as the torchxrayvision README

    rgba = np.dstack([rgb, np.full((20, 30), 7, dtype=np.uint8)])
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "rgba.png")
    assert np.allclose(read_xray(tmp_path / "rgba.png")[0], 30.0)  # alpha ignored

    Image.fromarray(np.zeros((5, 5), dtype=np.float32), mode="F").save(tmp_path / "f.tiff")
    with pytest.raises(ValueError, match="unsupported image mode"):
        read_xray(tmp_path / "f.tiff")


def test_centre_crop_keeps_the_middle_square():
    image = np.zeros((100, 300), dtype=np.uint8)
    image[:, 100:200] = 200  # only the central 100 x 100 square is bright
    out = preprocess_xray(image, 255)
    np.testing.assert_allclose(out["x224"], _normalised(200), atol=1e-3)


def test_public_images_preprocess():
    items = fixture_items(FIXTURES)
    assert len(items) == 4
    for _, path in items:
        out = preprocess_xray(*read_xray(path))
        x = out["x224"]
        assert x.shape == (1, 224, 224) and np.isfinite(x).all()
        assert x.min() >= -1024.01 and x.max() <= 1024.01 and x.std() > 10  # a real image, not blank
        assert out["display512"].shape == (512, 512)


def test_matches_the_torchxrayvision_readme_pipeline():
    import skimage.io
    import torchvision

    xrv = _xrv()
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    for _, path in fixture_items(FIXTURES):
        img = xrv.datasets.normalize(skimage.io.imread(path), 255)
        img = img.mean(2)[None, ...] if img.ndim == 3 else img[None, ...]
        reference = transform(img)
        ours = preprocess_xray(*read_xray(path))["x224"]
        np.testing.assert_allclose(ours, reference, atol=0.05)


def test_cache_is_resumable_and_round_trips(tmp_path):
    items = fixture_items(FIXTURES)
    first = build_cache(items, tmp_path)
    assert first.cached_now == len(items) and not first.missing and not first.failed
    second = build_cache(items, tmp_path)
    assert second.already_cached == len(items) and second.cached_now == 0
    for key, path in items:
        cached, fresh = load_cached(tmp_path, key), preprocess_xray(*read_xray(path))
        for name in ("x224", "display512"):
            assert np.array_equal(cached[name], fresh[name]), name
    (tmp_path / "broken.png").write_bytes(b"not an image")
    problems = build_cache([("gone", tmp_path / "gone.png"), ("broken", tmp_path / "broken.png")], tmp_path)
    assert problems.missing == ["gone"] and problems.failed == ["broken"]
    assert problems.counts()["missing_files"] == 1 and problems.counts()["failed_to_read"] == 1


def test_importing_torchxrayvision_keeps_our_warning_filters():
    # baseline after our module (numpy adds its own standard filters on import); a bare
    # `import torchxrayvision` would prepend ('ignore', None, Warning, ...) for every warning
    code = (
        "import warnings\n"
        "from fate_icu.data.cxr.preprocess import _xrv\n"
        "before = list(warnings.filters)\n"
        "_xrv()\n"
        "assert list(warnings.filters) == before, 'torchxrayvision changed the warning filters'\n"
        "import sys\n"
        "assert 'torchxrayvision' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)
