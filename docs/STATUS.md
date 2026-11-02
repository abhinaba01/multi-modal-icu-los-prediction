# Status

Build order (laptop): 0 → 2 → 1 → 5 → 3 → 4 → 6 → 7. The stage table is in `docs/stages.md`.
Each finished stage gets a summary of at most 10 lines here.

## Now

- **Phase:** build (laptop).
- **Stage 0:** built and verified on Python 3.11 with uv. 109 tests pass, including the 50
  legacy ones and the 3 `demo` tests; 2 `realdata` tests are skipped (lab only).
- **Smoke run:** `fate_icu.tabular.run` and `audit_leakage` run end to end on `mimiciv_demo`
  (140 ICU stays, 86 in the contract cohort).
- **Waiting for:** your review of `docs/contract.md`.
- **Lab, Stage 0** (`docs/lab_runbook.md`): load MIMIC-IV v3.1, run the tests including
  `realdata`, reproduce the internship with the legacy scripts, then freeze the
  long-stay threshold.

## Stage summaries

### Stage 0: repo and contract (build done, 2026-11-02)

- **Contract:** `configs/contract.yaml`, loaded by `fate_icu.data.contract`. The evidence is
  in `docs/contract.md`. The internship commit hash and metrics are lost; the snapshot is
  identified by `legacy_tree_sha256`.
- **Findings, reported:**
  - billed diagnoses are not available at t0 (excluded from FaTE inputs, D-001);
  - the internship split was stay-level and its Q3, encoding and scaling were fit on the
    whole cohort (D-002, D-003, D-007);
  - `storetime` is never checked (D-006);
  - the hand-rolled SOFA has defects (D-005).
- **Adapter:** `fate_icu.tabular`, which wraps the legacy code without editing it:
  - `load_cohort`, `load_static_features`, `load_splits`, `train_two_stage_xgb`, `explain_shap`;
  - PostgreSQL, CSV and in-memory loaders;
  - provenance for the no-leakage test.
- **Utilities:** settings from `.env`, the run manifest, seeding, device choice and
  small-count suppression.
- **Synthetic generator:** real MIMIC-IV schemas with planted cohort and leakage cases.
- **Docs:** `laptop_setup.md`, the Stage 0 runbook section, `decisions.md` (D-001–D-015),
  and the hypothesis and results-log templates.
- **Demo results (smoke only, not for reporting):** the 86-stay cohort splits 56/14/16
  into train/val/test; the storetime audit found 2-3 % of in-window rows stored after t0.
