# CLAUDE.md — FaTE-ICU (multi_modal_icu_los_prediction)

Part A holds the permanent rules. Stage 0 is below it. Stages 1–8 are in `docs/stages.md`; read only the stage I name.

# Part A — Core

## A1. What you are building

You are extending my M.Tech internship project: ICU length-of-stay (LOS) prediction on MIMIC-IV that replicates and extends Hempel et al. (2023) with a two-stage pipeline (classification, then regression on the stays predicted short). It extracts features with pandas from the raw MIMIC-IV CSVs (`sql/Extraction.sql` is a reference copy only) and fits four model families (logistic/linear regression, SVM, random forest, XGBoost), with random forest or XGBoost as the stage-1 prior. It adds a hand-rolled Python SOFA score with SOFA-trajectory, organ-failure-onset and sub-score-interaction feature families, and has 50 unit tests. Its code lives in `legacy_los/`; its frozen definitions are in `configs/contract.yaml` and `docs/contract.md`.

FaTE-ICU adds:
1. **Multimodal predictor:** the internship features + hourly vitals/labs + the patient's own chest X-ray from MIMIC-CXR.
2. **Attribution:** which inputs drove each prediction.
3. **Explanation:** MedGemma 1.5 4B, fine-tuned with QLoRA, explains each prediction using only the attributed drivers, citing literature retrieved by a RAG layer.
4. **Evaluation harness:** tests whether explanations are faithful (they name what actually moves the model) and grounded (the cited passages support them).

The thesis claim is the faithful, grounded explanation loop and its evaluation — not state-of-the-art LOS accuracy. If the X-ray adds little predictive value, we report that.

```
MIMIC-IV ─┬─ static features (internship) ──┐
          └─ hourly time series ────────────┼─► fusion model ─► P(long stay), LOS days
MIMIC-CXR ── chest X-ray taken before t0 ───┘           │
                                  attribution ─► evidence packet (drivers)
                RAG over PubMed, StatPearls, guidelines ─► passages
                MedGemma 1.5 4B + QLoRA ─► explanation with [D#] / [R#] citations
                evaluation harness: predictive · attribution faithfulness ·
                                    explanation faithfulness · groundedness
```

## A2. Two phases, two machines

- **Build phase — now, on my laptop (Windows, Intel CPU, no GPU, no MIMIC data).** You run only here. Write and test all code against the open MIMIC-IV demo, synthetic data and public data. Never assume CUDA. Give commands that work in PowerShell or cmd; don't assume bash or make. mimic-code's PostgreSQL build uses `psql` with its SQL scripts.
- **Run phase — later, on the college lab desktop (RTX A2000, 12 GB).** I clone the repo there and run real-data jobs from `docs/lab_runbook.md`. Each stage runs in the lab as soon as its code is built and its inputs exist; the two phases overlap.
- **Kaggle / Colab:** public, synthetic or MIMIC-IV-demo data only (e.g. embedding the PubMed corpus, smoke-testing QLoRA).
- **GCP (my credentialed project):** backup for lab jobs that need more than 12 GB, or when the lab is unavailable.

## A3. Data use rules — highest priority

PhysioNet's agreement forbids sharing credentialed data with third parties, including through APIs or online platforms. You run through an external API, so:
1. **Credentialed data** = MIMIC-IV, MIMIC-CXR and anything derived from them: features, tensors, image caches, embeddings, attributions, evidence packets, generated explanations, checkpoints. It exists only on the lab machine or in my GCP project — never on this laptop, Kaggle, Colab or git.
2. **You never see it.** Scripts that run on real data print aggregates only: counts, shapes, metrics, summary statistics. No IDs, rows, free text or images; suppress counts below 10. I paste those outputs back to you.
3. **On this laptop** the data is: synthetic fixtures (`tests/fixtures/`), the open-access MIMIC-IV Clinical Database Demo in a local PostgreSQL database `mimiciv_demo`, a handful of public chest X-rays, and public corpora. You may read all of these.
4. Real-database credentials live only in the lab machine's git-ignored `.env`.
5. `.gitignore` already covers `.env`, `data/`, `artifacts/`, `outputs/`, `reports/`, `checkpoints/`, `mlruns/`, `*.parquet`, `*.npz`, `*.npy`, `*.pt`, `*.safetensors`, and `*.csv` outside `tests/fixtures/`. `.claude/settings.json` denies reads on the same locations. Don't weaken either; if you think one needs changing, ask me.
6. `legacy_los/` is a read-only snapshot of the internship code (tag `v1.0-internship`): no data, no git history. I place it there myself before Stage 0. Never edit it.

## A4. Ground rules

- **The internship defines the contract.** Cohort, target, window W, long-stay threshold and patient split are recorded in `configs/contract.yaml` (Stage 0). Don't change them. If something looks wrong, stop and report it with evidence — the internship's SOFA SQL bugs are the precedent.
- **B0 stays faithful.** The internship baseline is reproduced exactly as in `legacy_los/` and Hempel et al.: no tuning, extra models, imputation or definition changes for speed or convenience. Deviations FaTE-ICU requires (patient-level split, train-only fitting, no post-t0 inputs) live in the adapter `src/fate_icu/tabular/` and in `docs/decisions.md`; the legacy scripts run as-is for B0-faithful.
- **No leakage.** t0 = ICU intime + W; every input must have been available at t0. Extraction functions return the latest source timestamp they used per stay; `tests/test_no_leakage.py` asserts it is ≤ t0.
- **Real pairings only.** Images are the patient's own studies; never assign images by LOS or any label-derived rule.
- **Patient-level splits**; every model arm is evaluated on the same test patients. Fit normalisation, imputation, thresholds and τ on train only.
- **Reproducibility:** every run writes `run_manifest.json` (git commit, config, seeds, data and package versions, model revisions); 5 seeds for neural models; nothing is tuned on test. Parallel execution must give results identical to serial; keep a guard test whenever you parallelise.
- **Fail loudly:** stop with a clear message when a required input is missing or a feature carries no signal; never fall back silently.
- **Null results are results:** log every experiment in `docs/results_log.md` with its statistical test.
- **Ask first** before changing a data definition, adding a dependency, deleting files, or running anything longer than 10 minutes on this laptop.

## A5. Code that runs on both machines

- **One codebase, two configs:** each pipeline has `configs/<stage>/smoke.yaml` (runs on this CPU in minutes, on demo or synthetic data) and `full.yaml` (lab). A stage isn't built until its smoke run passes end to end. Smoke results never back reported numbers.
- Paths, database URL and device come from `.env` (template: `.env.example`); nothing is hard-coded.
- Use `pathlib` and Python entry points (`python -m fate_icu...`), not shell scripts — the two machines may run different operating systems.
- Pick the device at runtime: `cuda` if available, else `cpu`.
- GPU-only packages (bitsandbytes, vLLM) go in an optional `gpu` extra and are imported lazily inside the functions that need them.
- Tests marked `gpu` or `realdata` are skipped on this laptop; plain `pytest` must pass here.
- Every stage adds a section to `docs/lab_runbook.md`: prerequisites, exact commands, expected runtime on the A2000, and which outputs to paste back to you.

## A6. Stack

Python 3.11, uv with a lock file, ruff, pytest, pre-commit · PostgreSQL with the internship's schema names (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`) · PyTorch 2.x · pandas or polars, pyarrow, pydantic, YAML configs · torchxrayvision, Pillow · captum, time_interpret (`tint`), grad-cam · transformers, peft, trl, accelerate (bitsandbytes and vLLM in the `gpu` extra) · bm25s, faiss-cpu, MedCPT · scikit-learn, scipy, statsmodels · MLflow (local) or TensorBoard. The APIs of tint, TRL and torchxrayvision change between releases — check the installed version before coding against them.

## A7. Repository layout

```
multi_modal_icu_los_prediction/
├── CLAUDE.md
├── .claude/settings.json    # read-deny rules for data locations
├── .env.example             # copied to a git-ignored .env on each machine
├── configs/                 # contract.yaml, variables.yaml, <stage>/{smoke,full}.yaml
├── legacy_los/              # internship pipeline at tag v1.0-internship (read-only; I copy it in)
├── sql/                     # new SQL: linkage, hourly time series
├── src/fate_icu/            # data/ tabular/ models/ attribution/ rag/ llm/ eval/ utils/
├── scripts/                 # thin CLI entry points
├── tests/                   # fixtures/: synthetic data + a few public chest X-rays
├── docs/                    # stages, STATUS, contract, decisions, hypotheses,
│                            # results_log, lab_runbook, laptop_setup,
│                            # internship_notes, internship_metrics
└── notebooks/               # exploration only; nothing is imported from here
```

## A8. Terms

- **W, t0:** prediction window in hours; prediction time t0 = ICU intime + W.
- **Paired cohort:** stays with an eligible chest X-ray taken before t0.
- **Evidence unit:** anything we can remove and re-score — a variable over a 4-hour block, a static feature group, the whole image, an image concept.
- **Driver:** an evidence unit selected as important for one prediction, with a direction and an effect size.
- **Evidence packet:** JSON holding the prediction and its drivers; the only patient information the LLM ever sees.
- **Faithful / grounded:** the explanation names what actually moves the model (tested by deletion) / each medical claim is supported by the passage it cites.
- **Smoke / full:** tiny CPU config on demo or synthetic data / real config for the lab.

## A9. How to work with me

- When I say **"Start Stage N"**: read that stage (Stage 0 below; Stages 1–8 in `docs/stages.md`), `docs/STATUS.md` and `docs/decisions.md`, then propose a plan (files, functions, tests, runbook commands) and wait for my approval.
- Tests first where practical → implementation → smoke run → update `docs/STATUS.md` and the runbook.
- Record design decisions in `docs/decisions.md`: date, decision, alternatives, reason.
- Core logic lives in `src/`, never in notebooks.
- When a stage's build is done, write a ≤10-line summary (artifacts, decisions, results) into `docs/STATUS.md`, tick the stage in the status table in `docs/stages.md`, and update A10.
- Commit after each stage with a clear message. Don't push without asking.

## A10. Current status

Phase: build (laptop). Stage 0 build done (the user's contract review is pending). Stages 2 and 1 built: 210 tests pass and both smoke runs work. Next: Stage 5.

---

# Stage 0 — Set up the repo and freeze the contract

**Goal:** a new repo containing the internship pipeline, its definitions frozen as a contract, and a working laptop environment.

**Build now (laptop):**
1. `legacy_los/` already contains the internship code at tag `v1.0-internship` (I copied it in). Don't edit it. The commit hash and my original test metrics are in `docs/internship_metrics.md`; copy them into `docs/contract.md`.
2. The internship's old notes are at `docs/internship_notes.md` (I moved them out of `legacy_los/`, because a CLAUDE.md there would load whenever you read files there and could contradict this file). If any CLAUDE.md remains under `legacy_los/`, tell me. List the rules in the notes that should move into Part A and show me before editing this file.
3. Make the legacy tests run here. Tests that need real data get the `realdata` marker; the rest must pass on `mimiciv_demo` or fixtures.
4. Read the code and write `docs/contract.md` and `configs/contract.yaml`:
   - SQL schema names; MIMIC-IV version.
   - Cohort rules: first ICU stay only? age limits? stays that end before t0?
   - Target: total or remaining LOS, units, the short/long threshold.
   - W, and which timestamp the features use (`charttime` or `storetime`).
   - Feature list in named groups (demographics, admission, SOFA, SOFA trajectory, care response, …), each with a plain-English label.
   - Split method, seed, and whether it is patient-level.
   - Check every feature's time window against W. If any feature uses data after t0, stop and tell me.

   Load the contract through a pydantic model in `src/fate_icu/data/contract.py`; later code imports it instead of hard-coding values.
5. `src/fate_icu/tabular/`: thin adapter exposing `load_cohort()`, `load_static_features()`, `load_splits()`, `train_two_stage_xgb()` and `explain_shap()`, without editing `legacy_los/`.
6. Laptop environment (Windows): give me the steps to load the MIMIC-IV demo into the local PostgreSQL database `mimiciv_demo` with the official mimic-code build scripts, and to build its derived concepts (including `sofa`) with mimic-code's PostgreSQL concept scripts. Then set up `.env.example`, the uv project with its `gpu` extra, pytest markers, pre-commit, the run-manifest and seeding utilities, a synthetic fixture generator that mirrors the real table schemas, and the `docs/` files listed in A7 (`STATUS.md`, `decisions.md`, `results_log.md`, `lab_runbook.md`; `stages.md` and `internship_metrics.md` already exist).
7. Create `docs/hypotheses.md` as a template with sections H1–H5 to be filled in during Stage 7 (see `docs/stages.md`, Stage 7).

**Run later (lab):**
- Load full MIMIC-IV into PostgreSQL with the same scripts and schema names; build the derived concepts.
- Run all legacy tests, including `realdata` ones; reproduce the internship's test metrics (`docs/internship_metrics.md`) within seed noise.

**Done when (build):** `pytest` passes on the laptop; I have reviewed the contract; the Stage 0 runbook section exists.
