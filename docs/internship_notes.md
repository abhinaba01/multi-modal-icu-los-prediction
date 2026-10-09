# Project context / handoff — Hempel et al. ICU-LOS replication

This file is auto-loaded by Claude Code so a fresh session (e.g. on another machine)
has the full context. **Read `README.md` for the complete, authoritative details** —
this is the short version.

## What this project is
A **faithful Python replication** of Hempel, Sadeghi, Kirsten (2023), *"Prediction of
Intensive Care Unit Length of Stay in the MIMIC-IV Dataset"* (Appl. Sci. 13, 6930).
- **Source of truth:** the paper (`applsci-13-06930.pdf`).
- **Code reference:** the authors' R code, GitLab
  `ul-mds/data-science/length-of-stay-prediction-mimic-iv`.
- The sibling `icu_los_prediction/` project is **NOT faithful** and is only a source of
  raw data / ideas.

## The prime directive
**Faithfulness to the paper + GitLab R code wins over everything** (speed, convenience).
If a change would deviate even slightly from the paper/R-code behaviour, do not make it.
All faithfulness decisions are documented in `README.md` (a "Resolved conflicts" table and
an "R → Python translation notes" section). Key ones: 18 labs (no phosphate), R²=corr²,
min-max scaling, label-encoding, complete-case (no imputation), Q3 short/long threshold,
LOS 1–21 days, 48h-readmission + death-in-ICU exclusions, and models matched to their
**R-library** defaults (SVM `gamma=1/p` + `scale=TRUE` incl. target; RF-regressor
`mtry=p/3`, `nodesize=5`; LogisticRegression unregularized; RF 1000 trees; XGB 3000 rounds;
no tuning; 10 repeated 80/20 splits).

## How to run (order matters)
```bash
pip install -r requirements.txt
python scripts/build_features.py --which all      # ALREADY DONE (parquets in data/processed/)
python scripts/run_hempel.py                       # writes results/hempel/ (+ per-repeat metrics)
python scripts/run_hempel_sofa.py
python scripts/run_hempel_sofa_dynamics.py
python scripts/run_ablation.py                     # reuses the 3 runs' per-repeat metrics; seconds
```
- Run from the project root. `run_ablation` must come **after** all three run scripts.
- Do **not** change `config.yaml` between the run scripts and the ablation (splits must match;
  there is a guard).

## Data the run scripts need
Only `data/processed/*.parquet` (the built feature tables, ~4 MB each). The 6.2 GB raw MIMIC
CSVs and the ICD GEM file are needed **only** by `build_features.py`, which is already done —
so they do not need to be present to run the models.

## Performance (faithfulness-neutral, results unchanged)
- Models fit in a **hybrid parallel** scheme (`src/evaluation/experiment.py`): SVM/SVR fits
  run concurrently across cores (they're single-threaded), while RF/XGB run one at a time
  using all cores. Controlled by `compute.n_jobs` in `config.yaml` (default `-1` = all cores;
  auto-adapts to any core count — no code change between 12- and 24-core machines).
- `tests/test_parallel_faithful.py` asserts parallel results == serial results (bit-identical).
- SVM `cache_size` is raised (speed only, solution unchanged).

## Status
Code complete; 50 tests pass (`python -m pytest tests/`). Feature tables built and verified
on MIMIC-IV **v3.1** (cohort 55,394; Q3=4.11 d; 24.9% long — matches the paper; the paper used
v2.1, so the absolute cohort is larger — the one unavoidable, documented deviation). The three
run scripts + ablation have not yet been run to completion at full faithful settings.

## When editing
- Keep `compute.n_jobs`-driven parallelism results identical (the faithful guard test must pass).
- Never trade faithfulness for speed. Do not add models, tune hyperparameters, impute, or change
  the cohort/feature/encoding/metric definitions.
