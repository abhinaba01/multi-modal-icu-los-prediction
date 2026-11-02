"""Hourly grid and the ``values`` / ``mask`` / ``delta`` tensors (Stage 2).

**Bins.** Hour ``h`` (``0 <= h < W``) holds times in ``(intime + h, intime + h + 1]``, so a
value exactly at t0 is in the last hour and one exactly at intime is outside, as in the
internship's ``(intime, t0]`` means. Several values in one hour are combined by each
variable's ``hourly`` rule; a vasopressor hour takes the highest dose of any interval
overlapping it.

**Tensors**, each ``[N, W, C]`` float32, per channel in this order:

1. ``mask`` = 1 where the hour has an accepted value (the raw grid is not NaN);
2. transform (log1p where set), then clip to the train 0.5-99.5th percentiles;
3. ``delta`` = hours since the last observation (0 when observed; ``h + 1`` before the
   first one), capped;
4. median channels: forward fill within the stay (never backward), then the train median
   with mask 0; zero channels (vasopressor): 0 when absent, never forward-filled;
5. standardise with the train mean and standard deviation.

Every statistic is fitted on observed cells of train stays only (CLAUDE.md A4).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fate_icu.data.timeseries.extract import Extraction
from fate_icu.data.timeseries.variables import VariableSet

ARRAYS = ("values", "mask", "delta")


def assign_bins(times: pd.Series, intimes: pd.Series, hours: int) -> np.ndarray:
    """Hour index of each time (``-1`` outside ``(intime, intime + hours]``)."""
    seconds = (pd.to_datetime(times) - pd.to_datetime(intimes)).dt.total_seconds().to_numpy()
    inside = (seconds > 0) & (seconds <= hours * 3600)
    bins = np.ceil(seconds / 3600.0) - 1
    return np.where(inside, bins, -1).astype(int)


def hourly_grid(
    extraction: Extraction, cohort: pd.DataFrame, variables: VariableSet, hours: int
) -> np.ndarray:
    """Raw hourly values in original units, ``[N, W, C]``, NaN where nothing was accepted.

    Args:
        extraction: Accepted measurements and intervals.
        cohort: Index ``stay_id`` (rows in this order); column ``intime``.
        variables: Channel order and hourly rules.
        hours: The window W.
    """
    cohort = cohort.sort_index()
    position = pd.Series(np.arange(len(cohort)), index=cohort.index)
    intime = pd.to_datetime(cohort["intime"])
    grid = np.full((len(cohort), hours, len(variables.variables)), np.nan)

    events = extraction.measurements
    if not events.empty:
        events = events[events["stay_id"].isin(cohort.index)]
        bins = assign_bins(events["time"], events["stay_id"].map(intime), hours)
        events = events.assign(pos=events["stay_id"].map(position).to_numpy(), bin=bins)
        if (events["bin"] < 0).any():
            raise RuntimeError("an accepted measurement lies outside the window")
        for c, variable in enumerate(variables.variables):
            sub = events[events["channel"] == variable.name]
            if sub.empty:
                continue
            agg = sub.groupby(["pos", "bin"])["value"].agg(variable.hourly)
            pos = agg.index.get_level_values("pos").to_numpy()
            hour = agg.index.get_level_values("bin").to_numpy()
            values = agg.to_numpy(dtype=float)
            if variable.floor is not None:
                values = np.maximum(values, variable.floor)
            grid[pos, hour, c] = values

    intervals = extraction.intervals
    if not intervals.empty:
        intervals = intervals[intervals["stay_id"].isin(cohort.index)]
        start_h = (
            pd.to_datetime(intervals["starttime"]) - intervals["stay_id"].map(intime)
        ).dt.total_seconds() / 3600
        end_h = (
            pd.to_datetime(intervals["endtime"]) - intervals["stay_id"].map(intime)
        ).dt.total_seconds() / 3600
        first = np.clip(np.floor(start_h.to_numpy()), 0, hours - 1).astype(int)
        last = np.clip(np.ceil(end_h.to_numpy()) - 1, 0, hours - 1).astype(int)
        pos = intervals["stay_id"].map(position).to_numpy()
        channel = intervals["channel"].map({v.name: i for i, v in enumerate(variables.variables)}).to_numpy()
        dose = intervals["value"].to_numpy(dtype=float)
        lengths = np.maximum(last - first + 1, 0)
        rows = np.repeat(np.arange(len(intervals)), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        hour = first[rows] + offsets
        # highest dose per hour; fmax ignores the NaN of hours not yet filled
        np.fmax.at(grid, (pos[rows], hour, channel[rows]), dose[rows])
    return grid


def _transform(raw: np.ndarray, variables: VariableSet) -> np.ndarray:
    x = raw.astype(float).copy()
    for c, variable in enumerate(variables.variables):
        if variable.transform == "log1p":
            x[:, :, c] = np.log1p(x[:, :, c])
    return x


@dataclass
class ChannelStats:
    """Train-fitted statistics of one channel, in transformed units."""

    name: str
    clip_low: float
    clip_high: float
    median: float
    mean: float
    std: float
    fill: float
    n_train_observed: int


@dataclass
class Normaliser:
    """Per-channel statistics fitted on train stays; applied to every split identically."""

    channels: list[ChannelStats]
    quantiles: tuple[float, float]
    delta_cap_hours: int
    warnings: list[str]

    @classmethod
    def fit(
        cls,
        raw: np.ndarray,
        train: np.ndarray,
        variables: VariableSet,
        quantiles: tuple[float, float] = (0.005, 0.995),
        delta_cap_hours: int = 24,
        require_signal: bool = True,
    ) -> Normaliser:
        """Fit on the train rows of the raw grid.

        Args:
            raw: ``[N, W, C]`` raw hourly values (NaN = unobserved).
            train: Boolean ``[N]``, True for train stays.
            variables: Channel definitions (transform, imputation).
            quantiles: Lower and upper clipping quantiles.
            delta_cap_hours: Cap for ``delta``.
            require_signal: Raise if a channel has no observed train value or is constant
                (CLAUDE.md A4, fail loudly); otherwise record a warning and use neutral stats.
        """
        x = _transform(raw[np.asarray(train, dtype=bool)], variables)
        stats, notes = [], []
        for c, variable in enumerate(variables.variables):
            cells = x[:, :, c]
            observed = cells[~np.isnan(cells)]
            fill = 0.0 if variable.impute == "zero" else np.nan
            if observed.size == 0:
                message = f"{variable.name}: no observed value in the train split"
                if require_signal:
                    raise ValueError(message)
                notes.append(message)
                stats.append(ChannelStats(variable.name, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0))
                continue
            low, high = (float(q) for q in np.quantile(observed, quantiles))
            clipped = np.clip(observed, low, high)
            median = float(np.median(clipped))
            if variable.impute == "zero":
                pool = np.where(np.isnan(cells), fill, np.clip(cells, low, high)).ravel()
            else:
                pool = clipped
            mean, std = float(np.mean(pool)), float(np.std(pool))
            if not np.isfinite(std) or std == 0.0:
                message = f"{variable.name}: constant in the train split"
                if require_signal:
                    raise ValueError(message)
                notes.append(message)
                std = 1.0
            stats.append(
                ChannelStats(
                    variable.name,
                    low,
                    high,
                    median,
                    mean,
                    std,
                    fill if variable.impute == "zero" else median,
                    int(observed.size),
                )
            )
        for note in notes:
            warnings.warn(note, stacklevel=2)
        return cls(stats, tuple(quantiles), int(delta_cap_hours), notes)

    def transform(self, raw: np.ndarray, variables: VariableSet) -> dict[str, np.ndarray]:
        """``values``, ``mask`` and ``delta`` (float32, ``[N, W, C]``) from a raw grid."""
        if [s.name for s in self.channels] != variables.names:
            raise ValueError("the normaliser was fitted on different channels")
        mask = ~np.isnan(raw)
        x = _transform(raw, variables)
        n, w, c = raw.shape
        delta = np.zeros((n, w, c))
        for t in range(w):
            if t == 0:
                delta[:, 0] = np.where(mask[:, 0], 0.0, 1.0)
            else:
                delta[:, t] = np.where(mask[:, t], 0.0, delta[:, t - 1] + 1.0)
        delta = np.minimum(delta, self.delta_cap_hours)

        for k, (variable, stats) in enumerate(zip(variables.variables, self.channels, strict=True)):
            channel = np.clip(x[:, :, k], stats.clip_low, stats.clip_high)  # NaN stays NaN
            if variable.impute == "median":
                for t in range(1, w):  # forward fill within each stay, never backward
                    gap = np.isnan(channel[:, t])
                    channel[gap, t] = channel[gap, t - 1]
            channel = np.where(np.isnan(channel), stats.fill, channel)
            x[:, :, k] = (channel - stats.mean) / stats.std
        return {
            "values": x.astype(np.float32),
            "mask": mask.astype(np.float32),
            "delta": delta.astype(np.float32),
        }

    def to_dict(self) -> dict:
        return {
            "quantiles": list(self.quantiles),
            "delta_cap_hours": self.delta_cap_hours,
            "warnings": list(self.warnings),
            "channels": [asdict(s) for s in self.channels],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Normaliser:
        return cls(
            [ChannelStats(**s) for s in data["channels"]],
            tuple(data["quantiles"]),
            int(data["delta_cap_hours"]),
            list(data.get("warnings", [])),
        )


def tensor_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash of the tensors (identical builds give identical hashes on any machine)."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
