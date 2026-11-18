"""Synthetic ``StayData`` with a learnable signal, for the Stage 3 model tests."""

from __future__ import annotations

import numpy as np

from fate_icu.models.data import StayData

N_STATIC, N_INDICATORS, HOURS, CHANNELS, IMAGE_DIM = 6, 2, 24, 5, 12
GROUPS = {"demographics": [0, 1], "vitals": [2, 3], "labs": [4, 5]}
INDICATOR_COLUMNS = [1, 4]


def random_stay_data(n: int = 120, seed: int = 0, signal: bool = True, image_share: float = 0.6) -> StayData:
    """``n`` stays; long stays depend on static column 0, a time-series channel and the image."""
    rng = np.random.default_rng(seed)
    static = rng.normal(size=(n, N_STATIC)).astype(np.float32)
    values = rng.normal(size=(n, HOURS, CHANNELS)).astype(np.float32)
    mask = (rng.random((n, HOURS, CHANNELS)) < 0.7).astype(np.float32)
    delta = rng.random((n, HOURS, CHANNELS)).astype(np.float32)
    image = rng.normal(size=(n, IMAGE_DIM)).astype(np.float32)
    has_image = rng.random(n) < image_share
    image[~has_image] = 0.0
    score = static[:, 0] + values[:, :, 0].mean(axis=1) * 3 + np.where(has_image, image[:, 0], 0.0)
    if not signal:
        score = rng.normal(size=n)
    y_long = (score > np.quantile(score, 0.7)).astype(np.float32)
    los = np.where(y_long > 0, 6.0, 2.0) + rng.random(n).astype(np.float32)
    # patients with one or two stays; splits by patient so no patient crosses a split
    patients = np.repeat(np.arange(n // 2 + 1), 2)[:n]
    unit = (patients * 0.6180339887) % 1.0
    split = np.where(unit < 0.7, "train", np.where(unit < 0.8, "val", "test"))
    return StayData(
        stay_ids=np.arange(1000, 1000 + n, dtype=np.int64),
        subject_ids=patients.astype(np.int64),
        split=split.astype(str),
        static=static,
        static_missing=(rng.random((n, N_INDICATORS)) < 0.2).astype(np.float32),
        values=values,
        mask=mask,
        delta=delta,
        image=image,
        has_image=has_image,
        y_long=y_long,
        los_days=los.astype(np.float32),
        static_groups=GROUPS,
        static_fill=np.zeros(N_STATIC, dtype=np.float32),
        indicator_columns=INDICATOR_COLUMNS,
        channels=[f"c{i}" for i in range(CHANNELS)],
        threshold_days=4.0,
    )
