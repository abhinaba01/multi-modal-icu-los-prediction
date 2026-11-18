"""Model inputs: one aligned set of arrays per stay (Stage 3).

``StayData`` holds, for the same stays in the same order:

* ``static``: the internship features, encoded as in B0 (categorical codes, then min-max)
  with every statistic fitted on **train stays only** (CLAUDE.md A4), missing values set to
  the train median, plus ``static_missing`` indicators for the columns that have a missing
  value in train;
* ``values`` / ``mask`` / ``delta``: the Stage 2 hourly tensors (already normalised on train);
* ``image``: the frozen image encoder's features per stay; ``has_image`` says whether the
  stay has an eligible X-ray (Stage 1). An absent image is zeros and is masked, never imputed;
* the targets: ``y_long`` and ``los_days``.

Images are only the patient's own studies (CLAUDE.md A4: real pairings only); this module
never assigns an image by any rule that sees a label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fate_icu.data.contract import Contract
from fate_icu.data.targets import long_stay_labels, round_los
from fate_icu.tabular.model import TabularPreprocessor

SPLITS = ("train", "val", "test")


@dataclass
class StaticEncoding:
    """Static feature encoding fitted on train stays (see the module docstring)."""

    columns: list[str]
    preprocessor: TabularPreprocessor
    fill: np.ndarray  # train median of each encoded column
    indicator_columns: list[int]  # encoded columns with a missing value in train
    groups: dict[str, list[int]]  # contract feature group -> encoded column positions
    constant_columns: list[str] = field(default_factory=list)  # no variation in train (smoke only)

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def n_inputs(self) -> int:
        return len(self.columns) + len(self.indicator_columns)

    @classmethod
    def fit(
        cls, X: pd.DataFrame, train: np.ndarray, contract: Contract, require_signal: bool = True
    ) -> StaticEncoding:
        """Fit on the train stays. A feature with no train value always stops the run; one with a
        single train value stops it unless ``require_signal`` is False (the smoke configs)."""
        columns = list(X.columns)
        if int(train.sum()) < 2:
            raise ValueError("need at least two train stays to fit the static encoding")
        raw = X[train]
        empty = [c for c in columns if raw[c].isna().all()]
        if empty:
            raise ValueError(f"static features with no train value: {empty}")
        constant = [c for c in columns if raw[c].nunique(dropna=True) <= 1]
        if constant and require_signal:
            raise ValueError(f"static features with a single train value (no signal): {constant}")
        pre = TabularPreprocessor(columns, contract.categorical(columns)).fit(X[train])
        encoded = pre.transform(X[train]).to_numpy(dtype=float)
        fill = np.nanmedian(encoded, axis=0)
        indicators = [int(j) for j in np.flatnonzero(np.isnan(encoded).any(axis=0))]
        groups: dict[str, list[int]] = {}
        for j, name in enumerate(columns):
            groups.setdefault(contract.feature(name).group, []).append(j)
        return cls(columns, pre, fill, indicators, groups, constant)

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(values, missing)``: values with gaps filled, and the 0/1 missing indicators."""
        encoded = self.preprocessor.transform(X[self.columns]).to_numpy(dtype=float)
        missing = np.isnan(encoded)
        values = np.where(missing, self.fill[None, :], encoded)
        return values.astype(np.float32), missing[:, self.indicator_columns].astype(np.float32)


@dataclass
class StayData:
    """Aligned arrays for the stays in ``stay_ids`` (see the module docstring)."""

    stay_ids: np.ndarray
    subject_ids: np.ndarray
    split: np.ndarray
    static: np.ndarray  # [N, F]
    static_missing: np.ndarray  # [N, K]
    values: np.ndarray  # [N, T, C]
    mask: np.ndarray
    delta: np.ndarray
    image: np.ndarray  # [N, D]
    has_image: np.ndarray  # [N] bool
    y_long: np.ndarray  # [N] float32, 0 or 1
    los_days: np.ndarray  # [N] float32
    static_groups: dict[str, list[int]] = field(default_factory=dict)
    static_fill: np.ndarray | None = None  # [F] value a removed static group takes
    indicator_columns: list[int] = field(default_factory=list)  # column behind each indicator
    channels: list[str] = field(default_factory=list)
    threshold_days: float = float("nan")

    def __post_init__(self) -> None:
        n = len(self.stay_ids)
        for name in (
            "subject_ids",
            "split",
            "static",
            "static_missing",
            "values",
            "mask",
            "delta",
            "image",
            "has_image",
            "y_long",
            "los_days",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        if self.mask.shape != self.values.shape or self.delta.shape != self.values.shape:
            raise ValueError("values, mask and delta must have the same shape")
        if np.any(self.image[~self.has_image] != 0):
            raise ValueError("a stay without an image must have an all-zero image feature")

    def __len__(self) -> int:
        return len(self.stay_ids)

    def rows(self, split: str | None = None, *, paired_only: bool = False) -> np.ndarray:
        """Row positions of a split (all rows when ``split`` is None), optionally paired only."""
        keep = np.ones(len(self), dtype=bool) if split is None else self.split == split
        if paired_only:
            keep &= self.has_image
        return np.flatnonzero(keep)

    @property
    def dims(self) -> dict[str, int]:
        return {
            "static": self.static.shape[1] + self.static_missing.shape[1],
            "ts_channels": self.values.shape[2],
            "hours": self.values.shape[1],
            "image": self.image.shape[1],
        }


def assemble(
    *,
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.Series,
    tensors: dict,
    image_features: dict[int, np.ndarray],
    image_dim: int,
    contract: Contract,
    threshold_days: float,
    require_signal: bool = True,
) -> tuple[StayData, StaticEncoding]:
    """Build ``StayData`` for the stays that have time-series tensors, in tensor order.

    Args:
        cohort: Contract cohort (index ``stay_id``) with ``subject_id`` and ``los``.
        features: Raw static features for the same stays.
        split: FaTE split label per stay.
        tensors: Output of ``fate_icu.data.timeseries.build.load_tensors``.
        image_features: ``stay_id -> feature vector`` for the stays with an eligible X-ray.
        image_dim: Length of one image feature vector.
        contract: The frozen contract.
        threshold_days: The long-stay threshold (frozen in the lab, train Q3 in smoke).
        require_signal: Stop when a static feature is constant in train (False for smoke runs).
    """
    stay_ids = np.asarray(tensors["stay_ids"], dtype=np.int64)
    absent = set(stay_ids.tolist()) - set(cohort.index.tolist())
    if absent:
        raise ValueError(f"{len(absent)} tensor stays are not in the cohort")
    order = pd.Index(stay_ids)
    cohort, features, split = cohort.loc[order], features.loc[order], split.loc[order]
    if not np.array_equal(split.to_numpy(dtype=str), np.asarray(tensors["split"], dtype=str)):
        raise ValueError("the split labels in the tensors differ from the contract split")
    encoding = StaticEncoding.fit(features, (split == "train").to_numpy(), contract, require_signal)
    static, static_missing = encoding.transform(features)
    unknown = set(image_features) - set(stay_ids.tolist())
    if unknown:
        raise ValueError(f"{len(unknown)} image features belong to stays outside the cohort")
    image = np.zeros((len(stay_ids), image_dim), dtype=np.float32)
    has_image = np.zeros(len(stay_ids), dtype=bool)
    for row, stay in enumerate(stay_ids):
        vector = image_features.get(int(stay))
        if vector is not None:
            if vector.shape != (image_dim,):
                raise ValueError(f"image feature has shape {vector.shape}, expected ({image_dim},)")
            image[row], has_image[row] = vector, True
    los = round_los(cohort["los"], contract).to_numpy(dtype=np.float32)
    y_long = long_stay_labels(cohort["los"], threshold_days, contract).to_numpy(dtype=np.float32)
    data = StayData(
        stay_ids=stay_ids,
        subject_ids=cohort["subject_id"].to_numpy(dtype=np.int64),
        split=split.to_numpy(dtype=str),
        static=static,
        static_missing=static_missing,
        values=np.asarray(tensors["values"], dtype=np.float32),
        mask=np.asarray(tensors["mask"], dtype=np.float32),
        delta=np.asarray(tensors["delta"], dtype=np.float32),
        image=image,
        has_image=has_image,
        y_long=y_long,
        los_days=los,
        static_groups=encoding.groups,
        static_fill=encoding.fill.astype(np.float32),
        indicator_columns=list(encoding.indicator_columns),
        channels=list(tensors["channels"]),
        threshold_days=float(threshold_days),
    )
    return data, encoding
