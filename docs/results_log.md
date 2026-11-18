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

