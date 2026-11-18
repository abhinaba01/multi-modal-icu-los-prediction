"""Stage 4 explainer: image drivers end to end (Grad-CAM, region, finding rule) and the demo run."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stay_data import random_stay_data

from fate_icu.attribution import image_cam, regions
from fate_icu.attribution import methods as M
from fate_icu.attribution.explain import Explainer, ImageContext
from fate_icu.attribution.gates import GatedLogit, build_layout
from fate_icu.attribution.packet import EvidencePacket
from fate_icu.attribution.select import Driver
from fate_icu.attribution.units import WHOLE_IMAGE, UnitSpace
from fate_icu.data.cxr.cache import fixture_items
from fate_icu.data.cxr.preprocess import build_cache
from fate_icu.models.fusion import FusionNet, ModelSpec
from fate_icu.models.image import FEATURE_WIDTH, load_image_encoder
from fate_icu.models.train import TrainedModel
from fate_icu.utils.env import REPO_ROOT

MODELS = Path.home() / ".torchxrayvision" / "models_data"
CHEX = MODELS / "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
PSP = MODELS / "pspnet_chestxray_best_model_4.pth"


@pytest.fixture(scope="module")
def explainer(tmp_path_factory):
    if not (CHEX.exists() and PSP.exists()):
        pytest.skip(f"download the chex and PSPNet weights first: {CHEX.name}, {PSP.name}")
    torch.manual_seed(0)
    encoder = load_image_encoder()
    cache = tmp_path_factory.mktemp("cxr")
    items = fixture_items(REPO_ROOT / "tests" / "fixtures" / "cxr")
    build_cache(items, cache)
    data = random_stay_data(n=40, seed=3, image_share=1.0)
    vectors = np.random.default_rng(0).normal(size=(40, encoder.dim)).astype(np.float32)
    effusion = encoder.pathology_names.index("Effusion")
    vectors[0, FEATURE_WIDTH + effusion] = 6.0  # stay 0: a strong effusion signal
    vectors[1, FEATURE_WIDTH + effusion] = -6.0  # stay 1: none
    data = dataclasses.replace(data, image=vectors)
    layout = build_layout(data, encoder.pathology_names, feature_width=FEATURE_WIDTH)
    net = FusionNet(
        ModelSpec(("static", "ts", "image"), "attention", d=16, heads=2, layers=1, dropout=0.0), data.dims
    )
    trained = TrainedModel(net.eval(), net.spec)
    keys = {int(s): items[i % len(items)][0] for i, s in enumerate(data.stay_ids)}
    return Explainer(
        trained=trained, model=GatedLogit(net, layout), space=UnitSpace(layout), data=data,
        raw=None, variables=None, contract=None, features=None,
        methods={"ts": "ig", "static": "ig", "concept": "ig"}, mcfg=M.MethodConfig(),
        image=ImageContext(encoder, cache, keys, image_cam.operating_thresholds(encoder)),
    )  # fmt: skip


def concept_driver(explainer, name, delta=0.5):
    unit = explainer.space.index[next(u for u in explainer.space.units if u.key == name)]
    return Driver((unit,), "image", name, delta)


def test_a_strong_concept_above_its_threshold_is_reported_with_a_region(explainer):
    detail = explainer._image(0, concept_driver(explainer, "Effusion"))
    assert detail["reported_finding"] is True
    assert (
        detail["label"].startswith("pleural effusion (image model estimate 1.00")
        or "image model estimate" in detail["label"]
    )
    allowed = {*regions.ZONES, regions.CARDIOMEDIASTINAL, regions.DIFFUSE}
    assert detail["region"] in allowed


def test_a_weak_concept_is_a_signal_not_a_finding(explainer):
    detail = explainer._image(1, concept_driver(explainer, "Effusion", delta=-0.4))
    assert detail["reported_finding"] is False
    assert detail["label"].startswith("image model signal for pleural effusion")
    assert "below its reporting threshold" in detail["label"]


def test_the_whole_image_has_no_finding_flag_but_may_have_a_region(explainer):
    unit = explainer.space.whole_image_index
    detail = explainer._image(2, Driver((unit,), "image", WHOLE_IMAGE, 0.3))
    assert "reported_finding" not in detail and detail["label"].startswith("chest X-ray")
    assert detail["region"] in {*regions.ZONES, regions.CARDIOMEDIASTINAL, regions.DIFFUSE}


def test_a_stay_without_a_film_gets_no_region(explainer):
    absent = dataclasses.replace(
        explainer.data,
        has_image=np.zeros(len(explainer.data), dtype=bool),
        image=np.zeros_like(explainer.data.image),
    )
    other = dataclasses.replace(explainer, data=absent)
    assert other._region(0, concept_driver(explainer, "Effusion")) is None


def test_drivers_need_tau_first(explainer):
    with pytest.raises(RuntimeError, match="fit tau"):
        explainer.drivers_for(np.array([0, 1]))


@pytest.mark.demo
def test_the_smoke_packets_are_valid_json_files_with_aggregate_only_output(tmp_path, monkeypatch, capsys):
    from fate_icu.attribution.build_packets import load_config, run

    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    config = load_config(REPO_ROOT / "configs/attribution/smoke.yaml").model_copy(
        update={"n_stays": 6, "ig_steps": 8, "mask_epochs": 20}
    )
    report = run(config)
    files = sorted((Path(report["out_dir"]) / "packets").glob("packet_*.json"))
    assert len(files) == 6
    for file in files:
        packet = EvidencePacket.model_validate_json(file.read_text(encoding="utf-8"))
        assert all(d.id == f"D{i}" for i, d in enumerate(packet.drivers, start=1))
    assert report["packets"] == "<10" and report["tau"] > 0
    text = json.dumps(report)
    assert not any(p.stem.split("_")[1] in text for p in files)  # no stay id in the report
