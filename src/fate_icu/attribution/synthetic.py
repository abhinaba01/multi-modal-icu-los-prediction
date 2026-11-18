"""Synthetic stays whose label depends on planted evidence units (Stage 4 benchmark).

A stay is long when at least one of four independent causes is present (each with probability
``prevalence``, plus a little label noise). Each cause is one evidence unit, so the units an
attribution method *should* find are known exactly:

====  ==========  =============================================================
Cause Modality    Planted unit
====  ==========  =============================================================
A     time series variable ``c2`` over hours 8-11 (a rise of +2.5 SD)
B     time series variable ``c4`` over hours 16-19 (a fall of -2.5 SD)
C     static      group ``labs`` (both of its features shifted +2.0)
D     image       concept ``k1`` (its logit shifted +3.0)
====  ==========  =============================================================

Everything else (the other variables, features, concepts, and the pooled image features) is
noise. The data are synthetic: no patient is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fate_icu.attribution.gates import GateLayout, build_layout
from fate_icu.attribution.units import UnitSpace
from fate_icu.models.data import StayData

CHANNELS = [f"c{i}" for i in range(6)]
GROUPS = {"demo": [0, 1], "vitals": [2, 3], "labs": [4, 5]}
INDICATOR_COLUMNS = [1, 4]
CONCEPTS = ["k0", "k1", "k2"]
FEATURE_WIDTH = 9
HOURS = 24
CAUSES = ("A", "B", "C", "D")


@dataclass
class Planted:
    data: StayData
    layout: GateLayout
    space: UnitSpace
    truth: np.ndarray  # [N, n_units] bool: the planted units active in each stay
    causes: np.ndarray  # [N, 4] bool: which of A-D are present


def planted_stays(
    n: int = 700, seed: int = 0, prevalence: float = 0.15, label_noise: float = 0.02
) -> Planted:
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.7, size=(n, HOURS, len(CHANNELS))).astype(np.float32)
    for t in range(1, HOURS):  # a little autocorrelation: hourly values are not independent
        values[:, t] = 0.5 * values[:, t - 1] + 0.87 * values[:, t]
    mask = (rng.random(values.shape) < 0.85).astype(np.float32)
    causes = rng.random((n, len(CAUSES))) < prevalence
    values[causes[:, 0], 8:12, 2] += 2.5
    values[causes[:, 1], 16:20, 4] -= 2.5
    values *= mask  # an unobserved hour carries no value
    delta = np.zeros_like(values)
    since = np.zeros((n, len(CHANNELS)), dtype=np.float32)
    for t in range(HOURS):
        since = np.where(mask[:, t] > 0, 0.0, since + 1.0 / HOURS)
        delta[:, t] = np.minimum(since, 1.0)
    static = rng.normal(size=(n, 6)).astype(np.float32)
    static[np.ix_(causes[:, 2], GROUPS["labs"])] += 2.0
    missing = (rng.random((n, len(INDICATOR_COLUMNS))) < 0.1).astype(np.float32)
    image = rng.normal(size=(n, FEATURE_WIDTH + len(CONCEPTS))).astype(np.float32)
    image[causes[:, 3], FEATURE_WIDTH + 1] += 3.0
    long = causes.any(axis=1) ^ (rng.random(n) < label_noise)
    los = np.where(long, 6.0, 2.0) + rng.random(n)
    unit = (np.arange(n) * 0.6180339887) % 1.0  # a split by patient (every stay its own patient)
    split = np.where(unit < 0.7, "train", np.where(unit < 0.8, "val", "test")).astype(str)
    data = StayData(
        stay_ids=np.arange(5000, 5000 + n, dtype=np.int64),
        subject_ids=np.arange(n, dtype=np.int64),
        split=split,
        static=static,
        static_missing=missing,
        values=values,
        mask=mask,
        delta=delta,
        image=image,
        has_image=np.ones(n, dtype=bool),
        y_long=long.astype(np.float32),
        los_days=los.astype(np.float32),
        static_groups=GROUPS,
        static_fill=np.zeros(6, dtype=np.float32),
        indicator_columns=INDICATOR_COLUMNS,
        channels=list(CHANNELS),
        threshold_days=4.0,
    )
    layout = build_layout(data, CONCEPTS, feature_width=FEATURE_WIDTH)
    space = UnitSpace(layout)
    truth = np.zeros((n, len(space)), dtype=bool)
    truth[causes[:, 0], space.ts_unit("c2", 2)] = True
    truth[causes[:, 1], space.ts_unit("c4", 4)] = True
    truth[causes[:, 2], space.n_ts + list(GROUPS).index("labs")] = True
    truth[causes[:, 3], space.concept_slice.start + 1] = True
    return Planted(data, layout, space, truth, causes)
