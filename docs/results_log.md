# Results log

Every experiment, including null results (CLAUDE.md A4). Each entry records:
- date, stage and run id (the `run_manifest.json` in the lab's outputs);
- config, data (MIMIC version, population) and seeds;
- result with CIs, the statistical test and its outcome;
- a one-line interpretation.

Smoke runs on the demo or synthetic data never appear here.

| Date | Stage | Run id | Config | Data / population | Seeds | Result (95 % CI) | Test | Interpretation |
|---|---|---|---|---|---|---|---|---|
| 2026-11-16 | 4 | attribution_benchmark_20261116T071826Z | configs/attribution/benchmark.yaml | synthetic planted units, 60 explained test stays | 666 | IG time series precision@1 0.95 (0.87-1.00), ExtremalMask 0.85 (0.72-0.95), DynaMask 0.21 (0.08-0.33) | bootstrap over stays (descriptive; no hypothesis test) | IG is the default; synthetic, so it does not show real-data performance |
| 2026-11-21 | 3 | models_first_pass_20261120T180632Z | configs/models/first_pass.yaml | MIMIC-IV v3.1 contract cohort, P2 test split (11,110 stays), no images | 666 (1) | AUROC B0 0.778 (0.768-0.788), B1 0.793 (0.783-0.803), B2 0.804 (0.794-0.814), B3 0.813 (0.803-0.823); AUPRC 0.574 / 0.586 / 0.614 / 0.626 | patient-level bootstrap CIs per arm (1,000 resamples); no paired test yet | Time series and static features add to the internship XGBoost; B3 is best on every metric; one seed, lambda untuned |
| 2026-11-21 | 3 | models_first_pass_20261120T180632Z (image arms added) | configs/models/image_arms.yaml (resume) | MIMIC-IV v3.1 + MIMIC-CXR-JPG, P1 = 2,360 paired test stays | 666 (1) | P1 AUPRC: M 0.670 (0.629-0.711), B3 0.674 (0.635-0.712), B4 (image only) 0.372 (0.332-0.418), B5 0.677, B6 0.676 | per-arm bootstrap CIs; paired H1 test NOT run (needs 5 seeds) | PRELIMINARY: the X-ray adds no visible predictive value on P1 in a one-seed look; fusion helps on P2 only through the architecture |

---

<!-- Entry template
### YYYY-MM-DD · Stage N · <run id>
- **Question:**
- **Config:** configs/<stage>/full.yaml @ <git commit>
- **Data:** MIMIC-IV v3.1, population P1/P2, n = …
- **Seeds:**
- **Result:**
- **Statistical test:**
- **Interpretation:**
-->

### 2026-11-16 · Stage 4 · attribution_benchmark_20261116T071826Z
- **Question:** which attribution method recovers planted evidence units, and is the explained model's answer
  supported by deleting them?
- **Config:** configs/attribution/benchmark.yaml (ExtremalMask settings tuned on seed 1234, evaluated on seed 666).
- **Data:** synthetic stays with four planted units (docs/attribution_benchmark.md); no MIMIC data. n = 60 explained
  test stays, test AUROC of the explained model 0.945.
- **Seeds:** 666 (one model, one seed).
- **Result:** time series, k = 3: IG precision@1 0.95 (0.87-1.00), recall@3 0.97; occlusion 0.95 / 0.94; ExtremalMask
  0.85 / 0.83; DynaMask 0.21 / 0.37. Comprehensiveness 0.59-0.61 for IG, occlusion and ExtremalMask, 0.04 for a random
  order. Final drivers: top driver planted 0.93-0.95, precision 0.51-0.53, recall 0.88-0.89.
- **Statistical test:** bootstrap over the explained stays for the intervals; no hypothesis test (a method
  comparison on 60 stays of one model is descriptive).
- **Interpretation:** IG and occlusion find the planted units; ExtremalMask is close only after tuning; DynaMask
  does not. Synthetic data cannot show real-data performance, and about half of the selected drivers are real but
  unplanted effects.

### 2026-11-21 · Stage 3 · models_first_pass_20261120T180632Z
- **Question:** before any image is used, how much do the hourly time series add to static features, and how does the
  EHR-only neural model compare with the internship's two-stage XGBoost (B0)?
- **Config:** configs/models/first_pass.yaml (d = 256, 2 layers, 50 epochs with patience 8, lambda_reg 1.0 not tuned),
  Stage 2 tensors `timeseries_full_20261120T163712Z` (tensor sha 5b86799f...), contract threshold 4.1 d (frozen).
- **Data:** MIMIC-IV v3.1, contract cohort 55,394 stays; train 38,874, val 5,410, test 11,110; 77 static inputs; no images.
- **Seeds:** 666 only (CLAUDE.md A4 asks for 5 in the final run).
- **Result (test, 95 % CI):**

  | Arm | AUROC | AUPRC | Brier | MAE days | RMSE days |
  |---|---|---|---|---|---|
  | B0 XGBoost two-stage | 0.778 (0.768-0.788) | 0.574 (0.553-0.597) | 0.178 | 1.885 | 3.431 |
  | B1 static MLP | 0.793 (0.783-0.803) | 0.586 (0.566-0.608) | 0.176 | 1.714 | 2.874 |
  | B2 time series LSTM | 0.804 (0.794-0.814) | 0.614 (0.595-0.635) | 0.181 | 1.749 | 2.790 |
  | B3 static + time series | 0.813 (0.803-0.823) | 0.626 (0.606-0.646) | 0.168 | 1.686 | 2.729 |

- **Statistical test:** per-arm bootstrap intervals over test patients; no paired comparison yet. The B2 and B3 intervals
  overlap, so their order is not established; B0 and B3 do not overlap but that is an unpaired comparison. The paired
  bootstrap with Holm correction (H1 and the grid) is part of Stage 7.
- **Interpretation:** the time series help (B2 above B1 on AUPRC), and the combination is best on every metric, in
  line with the Hempel et al. claim that EHR time-dependent features matter. One seed and an untuned lambda: treat the
  ordering as a first look. The image arms are the H1 question and are not run yet.

### 2026-11-21 · Stage 3 · models_first_pass_20261120T180632Z (image arms added; PRELIMINARY)
- **Question:** does the patient's own chest X-ray add predictive value (H1, registered 2026-11-21 before these numbers were read)?
- **Config:** configs/models/image_arms.yaml (resume of the first pass: same model size and recipe; arms B4, B5, B6, M, M_transformer
  added; images from the 11,498-stay Stage 1 link `cxr_full_20261120T193509Z`, DenseNet-121 `densenet121-res224-chex`).
- **Data:** MIMIC-IV v3.1 and MIMIC-CXR-JPG 2.1.0; train 38,874, val 5,410, test 11,110 stays; P1 = 2,360 test stays with an X-ray before t0.
- **Seeds:** 666 only. The registered H1 test uses 5-seed ensembles, so this is a preliminary look, not the registered test.
- **Result (test, 95 % CI, unpaired):**

  | Arm | P1 AUROC | P1 AUPRC | P2 AUROC | P2 AUPRC |
  |---|---|---|---|---|
  | B3 static + time series | 0.828 (0.807-0.848) | 0.674 (0.635-0.712) | 0.813 | 0.626 |
  | B4 image only | 0.636 (0.609-0.662) | 0.372 (0.332-0.418) | – | – |
  | B5 late fusion | 0.829 | 0.677 | 0.811 | 0.616 |
  | B6 | 0.830 | 0.676 | 0.819 | 0.634 |
  | M attention fusion | 0.827 (0.807-0.848) | 0.670 (0.629-0.711) | 0.819 | 0.635 |
  | M_transformer | 0.816 | 0.661 | 0.814 | 0.628 |

- **Statistical test:** none yet. H1 is the paired patient-level bootstrap of AUPRC(M) - AUPRC(B3) on P1 with Holm correction, on
  5-seed ensembles (`docs/hypotheses.md`).
- **Interpretation:** on the paired stays M is at the level of the EHR-only B3 (difference about -0.004 in AUPRC, intervals overlap
  almost entirely); the image alone is only weakly predictive (AUROC 0.64). If the 5-seed test agrees, H1 is not supported and
  the thesis reports that the X-ray adds little (CLAUDE.md A1). M's small gain on all test stays comes from the fusion
  architecture, since most of those stays have no image.

