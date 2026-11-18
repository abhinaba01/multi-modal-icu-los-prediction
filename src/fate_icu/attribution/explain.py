"""From a trained model and a stay to an evidence packet (Stage 4).

For each stay: score every evidence unit with the chosen methods (``methods.py``), take the
two-step driver selection (``select.py``), and turn each driver into a packet entry whose
text comes from the deterministic templates of ``packet.py``. Image drivers also get Grad-CAM
through the DenseNet and the anatomical region of the heatmap; a pathology concept is reported
as a finding only if its probability is above the image model's operating threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fate_icu.attribution import image_cam, regions
from fate_icu.attribution import methods as M
from fate_icu.attribution import packet as P
from fate_icu.attribution.gates import GatedLogit, stay_tensors
from fate_icu.attribution.select import Driver, SelectionConfig, candidate_units, fit_tau, select_drivers
from fate_icu.attribution.units import WHOLE_IMAGE, StayScorer, UnitSpace
from fate_icu.data.contract import Contract
from fate_icu.data.cxr.preprocess import load_cached
from fate_icu.data.timeseries.variables import VariableSet
from fate_icu.models.data import StayData
from fate_icu.models.image import FEATURE_WIDTH, ImageEncoder
from fate_icu.models.train import TrainedModel, predict


@dataclass
class ImageContext:
    """What image drivers need: the encoder, the preprocessed films and the region model."""

    encoder: ImageEncoder
    cache_dir: Path
    keys: dict[int, str]  # stay_id -> image-cache key
    thresholds: dict[str, float]
    segmenter: regions.Segmenter | None = None  # loaded on first use (273 MB)

    def get_segmenter(self) -> regions.Segmenter:
        if self.segmenter is None:
            self.segmenter = regions.load_segmenter()
        return self.segmenter


@dataclass
class Explainer:
    trained: TrainedModel
    model: GatedLogit
    space: UnitSpace
    data: StayData
    raw: np.ndarray  # [N, T, C] original units
    variables: VariableSet
    contract: Contract
    features: pd.DataFrame  # raw static features by stay_id
    methods: dict[str, str]  # time series / static / concept method names
    mcfg: M.MethodConfig
    sel: SelectionConfig = field(default_factory=SelectionConfig)
    tau: float | None = None
    image: ImageContext | None = None
    window_hours: int = 24

    def scores(self, rows: np.ndarray) -> np.ndarray:
        fixed = stay_tensors(self.data, rows)
        return M.attribute(
            self.model, self.space, fixed, self.mcfg,
            ts=self.methods["ts"], static=self.methods["static"], concept=self.methods["concept"],
        )  # fmt: skip

    def fit_tau(self, rows: np.ndarray, chunk: int = 32) -> float:
        """Fit ``tau`` on validation stays (never on test): see ``select.fit_tau``."""
        scorers, candidates = [], []
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            scores = self.scores(part)
            for i, row in enumerate(part):
                scorers.append(StayScorer(self.model, self.space, self.data, int(row)))
                candidates.append(candidate_units(scores[i], self.space, self.sel.per_modality))
        self.tau = fit_tau(scorers, candidates, self.sel, self.mcfg.seed)
        return self.tau

    def drivers_for(self, rows: np.ndarray, chunk: int = 32) -> list[tuple[int, list[Driver]]]:
        if self.tau is None:
            raise RuntimeError("fit tau on validation stays first (Explainer.fit_tau)")
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            scores = self.scores(part)
            for i, row in enumerate(part):
                scorer = StayScorer(self.model, self.space, self.data, int(row))
                out.append((int(row), select_drivers(scorer, scores[i], self.tau, self.sel)))
        return out

    # ---- turning drivers into packet entries ---------------------------------------------------

    def _timeseries(self, row: int, d: Driver) -> dict:
        variable = next(v for v in self.variables.variables if v.name == d.key)
        c = self.data.channels.index(d.key)
        return {
            "label": variable.label,
            "time": P.block_label(d.start_hour, d.end_hour),
            "observed": P.describe_timeseries(variable, self.raw[row, :, c], d.start_hour, d.end_hour),
            "reference_range": variable.reference_range.text,
        }

    def _static(self, row: int, d: Driver) -> dict:
        stay = int(self.data.stay_ids[row])
        values = self.features.loc[stay].to_dict()
        return {
            "label": self.contract.feature_groups[d.key].label,
            "observed": P.describe_static(self.contract, d.key, values),
        }

    def _image(self, row: int, d: Driver) -> dict:
        concepts = self.space.layout.concepts
        probability = None
        reported = False
        if d.key != WHOLE_IMAGE and self.image is not None:
            probs = image_cam.concept_probabilities(
                self.data.image[row], concepts, self.space.layout.feature_width
            )
            probability = probs[d.key]
            reported = image_cam.finding_allowed(probability, self.image.thresholds.get(d.key, float("nan")))
        out = {"label": P.describe_image(d.key, probability, reported)}
        if d.key != WHOLE_IMAGE:
            out["reported_finding"] = reported
        region = self._region(row, d)
        if region is not None:
            out["region"] = region
        return out

    def _region(self, row: int, d: Driver) -> str | None:
        """Anatomical region of the Grad-CAM heatmap (description only), if the film is available."""
        if self.image is None or not self.data.has_image[row]:
            return None
        key = self.image.keys.get(int(self.data.stay_ids[row]))
        if key is None:
            return None
        cached = load_cached(self.image.cache_dir, key)
        fixed = stay_tensors(self.data, np.array([row]))
        wrapper = image_cam.ImageToLogit(self.image.encoder, self.model, fixed)
        heat = image_cam.grad_cam(wrapper, cached["x224"], 1 if d.delta > 0 else -1)
        masks = self.image.get_segmenter().masks(cached["display512"])
        return regions.describe_region(regions.region_shares(heat, masks))

    def packet(self, row: int, drivers: list[Driver]) -> P.EvidencePacket:
        prediction = predict(self.trained, self.data, np.array([row]))
        entries = []
        for i, d in enumerate(drivers, start=1):
            detail = {"timeseries": self._timeseries, "static": self._static, "image": self._image}[
                d.modality
            ](row, d)
            effect = round(d.delta, 4) or d.delta
            entries.append(P.PacketDriver(
                id=f"D{i}", modality=d.modality, key=d.key, direction=d.direction, effect=effect, **detail
            ))  # fmt: skip
        return P.EvidencePacket(
            prediction=P.PacketPrediction(
                p_long_stay=round(float(prediction["p_long"][0]), 4),
                predicted_los_days=round(float(prediction["los_days"][0]), 2),
                long_stay_threshold_days=float(self.data.threshold_days),
                window_hours=self.window_hours,
            ),
            drivers=entries,
        )

    def packets(self, rows: np.ndarray) -> list[tuple[int, P.EvidencePacket]]:
        return [(row, self.packet(row, drivers)) for row, drivers in self.drivers_for(rows)]


def encoder_dims_check(data: StayData, encoder: ImageEncoder) -> None:
    """The stay data must hold the encoder's vectors: pooled features, then concept logits."""
    if data.image.shape[1] != FEATURE_WIDTH + len(encoder.pathology_index):
        raise ValueError("StayData image vectors do not match the image encoder")
