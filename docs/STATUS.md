

 # Status

Build order (laptop): 0 → 2 → 1 → 5 → 3 → 4 → 6 → 7. The stage table is in `docs/stages.md`.
Each finished stage gets a summary of at most 10 lines here.

## Now

- **Phase:** build (laptop).
- **Stage 0:** built and verified on Python 3.11 with uv. 109 tests pass, including the 50
  legacy ones and the 3 `demo` tests; 2 `realdata` tests are skipped (lab only).
- **Smoke run:** `fate_icu.tabular.run` and `audit_leakage` run end to end on `mimiciv_demo`
  (140 ICU stays, 86 in the contract cohort).
- **Stage 2:** built 2026-11-02. 160 tests pass (2 `realdata` skipped); the smoke build runs
  end to end on `mimiciv_demo`.
- **Stage 1:** built 2026-11-02. 210 tests pass (2 `realdata` skipped); the smoke link
  (demo cohort, synthetic metadata) and the cache step (public images) run end to end.
- **Stage 5:** code and smoke run done 2026-11-13. 364 tests pass (2 `realdata` skipped). The
  full index is pending: your `.env` settings and the guideline PDFs, then the downloads, the
  Kaggle GPU step and your relevance grades (`docs/lab_runbook.md`, Stage 5).
- **Waiting for:** your review of `docs/contract.md` (Stage 0); for Stage 5, `FATE_RAG_DIR`
  outside OneDrive and `NCBI_EMAIL` in `.env`, the four guideline PDFs with their licences,
  and your review of `configs/rag/pubmed_queries.yaml`.
- **Lab, Stage 0** (`docs/lab_runbook.md`): load MIMIC-IV v3.1, run the tests including
  `realdata`, reproduce the internship with the legacy scripts, then freeze the
  long-stay threshold.

## Stage summaries

### Stage 5: RAG layer (code and smoke run done, 2026-11-13; full index pending)

- **Corpus:** PubMed through E-utilities (7 topic and 37 driver searches), all of StatPearls
  (the 2.05 GB archive) and four guidelines, cut into passages of at most 300 MedCPT tokens
  within sections (D-033 to D-036).
- **Retrieval:** BM25, then MedCPT at pinned revisions in an exact FAISS index, fused with RRF
  and reranked by the cross-encoder; top 3 per driver as R1..Rn; ties by passage id (D-031,
  D-037). MedCPT reproduces its model-card outputs here.
- **Queries:** 100 fixed queries over 50 driver labels, run once into a query bank, so
  `retrieve(packet)` is a deterministic lookup. Stage 4 packets must carry each driver's `key`
  (D-038).
- **GPU step:** one command on Kaggle or Colab (embeddings, FAISS, bank), checked on the laptop
  before it is installed (D-039).
- **Evaluation:** blind pooled sheet, a terminal grader, P@3 and nDCG@5 with CIs and paired
  tests (D-040).
- **Smoke run** (synthetic fixtures, real MedCPT on this CPU, 404 s): 22 passages, every check
  passed.

### Stage 1: link MIMIC-IV and MIMIC-CXR (build done, 2026-11-02)

- **Code:** `fate_icu.data.cxr`, covering:
  - metadata read as text, with a StudyDate/StudyTime parser that keeps fractions;
  - synthetic metadata with 18 planted scenarios, keyed to any cohort;
  - the linkage, the download list, the flow report, preprocessing and the cache.
- **Selection:** frontal AP/PA, intime − 24 h ≤ study ≤ t0, the latest image, ties AP then
  `dicom_id`, every stay kept with `has_cxr`, the contract split (D-024 to D-026). It
  differs from MedFuse on every point (D-026).
- **Preprocessing:** torchxrayvision 1.5.7 normalisation, centre crop, 224 plus a 512
  display copy, cached per image. The import is guarded because torchxrayvision silences
  all warnings (D-027).
- **Smoke run:** all 86 demo stays got the correct planted answer across 18 scenarios; 71
  stays were paired. The 4 public images (D-028) are cached.
- **New dependencies:** torch 2.15, torchvision, torchxrayvision, scikit-image, pillow
  (D-029). The lab driver must support CUDA 13 by Stage 3.

### Stage 2: hourly time-series tensors (build done, 2026-11-02)

- **Channels:** `configs/variables.yaml` has 30 (18 with the internship's itemids, plus
  mimic-code's BP and blood-gas items and the official vasopressor and per-hour SOFA
  concepts). Each has a label, unit, synonyms and reference range.
- **Code:** `fate_icu.data.timeseries`: SQL (`sql/timeseries/`) or in-memory sources, then
  the availability and plausibility rules with provenance, the hourly grid, a train-fitted
  normaliser, and the `values`, `mask`, `delta` and `raw` arrays.
- **Rules:** the window is the internship's (intime, t0]; rows need `storetime` ≤ t0; GCS
  follows the internship (ETT = verbal 1); urine subtracts irrigant volume in; the
  vasopressor channel is 0 when no infusion runs (D-016 to D-023).
- **Two more internship findings** in `docs/contract.md` §7: irrigant counted as urine, and
  intubated patients getting low GCS.
- **Demo smoke build** (`[86, 24, 30]`, about 10 s): 2.9 % of chart rows were stored after t0;
  heart-rate means differ from the internship's by a median of 0.29 bpm.

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
