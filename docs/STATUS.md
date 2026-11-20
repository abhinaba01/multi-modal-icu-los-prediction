

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
- **Stage 5:** code and smoke run done 2026-11-13. Corpus `17a268f4f611` (426,770 passages) and
  the BM25 index are built (2026-11-15). Still needed: the Kaggle GPU step, the dense index and
  your relevance grades (`docs/lab_runbook.md`, Stage 5).
- **Stage 3:** built 2026-11-16. 491 tests pass (2 `realdata` skipped, 1 `gpu` skipped); the smoke
  run trains every arm end to end on the demo. Smoke numbers mean nothing (16 test stays).
- **Stage 4:** built 2026-11-16 (D-048). The packet run works end to end on the demo (16 valid packets); the
  synthetic benchmark (`docs/attribution_benchmark.md`) is reviewed; your default is ExtremalMask for the time series, IG elsewhere (open: a
  relative-size condition on drivers). Not built: the semi-synthetic
  benchmark on real tensors and the validation report on the real model (lab).
- **Run machine (2026-11-20, D-049):** the lab is not available, so all real jobs run on a GCP L4 spot VM
  (runbook section "GCP: the run machine"). Stage 3 saves each arm as it is trained and resumes with
  `--resume <run id>` (`tests/test_models_resume.py`). The laptop's Python was blocked by Windows Application
  Control until the environment was rebuilt from a python.org 3.11 (2026-11-20); 588 tests now pass (3 skipped).
- **Waiting for:** your review of `docs/contract.md` (Stage 0); for Stage 4, your decision on the relative-size condition on drivers; for Stage 5, the Kaggle GPU step,
  a browser spot-check of one StatPearls link (D-043), the open StatPearls questions (D-043) and
  your review of `configs/rag/pubmed_queries.yaml`.
- **Lab, Stage 0** (`docs/lab_runbook.md`): load MIMIC-IV v3.1, run the tests including
  `realdata`, reproduce the internship with the legacy scripts, then freeze the
  long-stay threshold.

## Resume here: Stage 5 (updated 2026-11-15, on the new computer)

State: the project moved to the new computer and passes there (406 tests, 2 `realdata`
skipped; the demo database is restored; `D:\fate-data` checks ok). The public data is in
`D:\fate-data\rag\raw` (StatPearls, 151,341 PubMed records, the four guideline PDFs). The
full index is not built. What remains, in order:

1. ~~**Guideline cut points.**~~ Done 2026-11-15 (D-045): the cut points are in
   `configs/rag/full.yaml`, three heading rules were added after reading the parsed PDFs, and
   the new fields and rules have tests (47 guideline tests). Table cells sometimes still
   become section titles (known limit in D-045).
2. ~~**Licences**~~: done 2026-11-15 (D-046). Recorded as each PDF states them; the
   guidelines check reports `ready: true`. Earlier notes: the user had to confirm them before
   the build would run (it stops while one says "TO CONFIRM"). What each PDF states: `ssc2021` "(c) 2021 ESICM and SCCM", no open
   licence in the PDF; `kdigo_aki_2012` "(c) 2012 KDIGO", none; `esicm_ards_2023` Creative
   Commons Attribution-NonCommercial (Crossref: CC BY-NC 4.0); `padis_2018` "(c) 2018 SCCM and
   Wolters Kluwer Health, all rights reserved".
3. **Build.** Laptop part done 2026-11-15 (runbook 5.2):
   - `build_corpus`: corpus `17a268f4f611` with 426,770 passages (PubMed 265,346 from
     151,126 documents, StatPearls 160,640 from 9,651 articles, guidelines 784 from 4).
     Tokens per passage: median 203, 95th percentile 285, max 300. 14 documents gave no
     passage.
   - `build_index --part bm25`: index version `b2112c8c296424a0`, vocabulary 160,570; every
     query has BM25 hits.
   - `gpu_bundle`: 4 files in `D:\fate-data\rag\rag_full\gpu_bundle` (passages.parquet
     171 MB).

   **Next:** the Kaggle GPU step (runbook 5.3, 1 to 2 hours on a T4). Put its `gpu_out` into
   `D:\fate-data\rag\rag_full\index\b2112c8c296424a0\gpu_out`, then run
   `build_index --part dense` and `verify` (5.4). After that, the user grades the pooled
   passages.
4. **Open questions for the user:** keep or restore the dropped StatPearls section "Enhancing
   Healthcare Team Outcomes", and the 20-token minimum (D-043); open
   `https://www.ncbi.nlm.nih.gov/books/n/statpearls/article-22481/` in a browser, which should
   show "Anatomy, Hair".

Afterwards: Stage 3 (docs/stages.md).

## Stage summaries

### Stage 4: attribution and evidence packets (build done, 2026-11-16)

- **Code:** `fate_icu.attribution` (gates, units, methods, select, image_cam, regions, packet, explain,
  build_packets, benchmark, synthetic) and `fate_icu.eval.attr_faithfulness`; configs in `configs/attribution/`.
- **Idea:** every method works on presence gates whose 0 is exactly the removal the model was trained with, so
  occlusion, Integrated Gradients, the masks and the driver-selection delta measure the same thing (D-048).
- **Selection:** top 5 per modality, then exact removal delta against tau (validation), merge, at most 5 drivers.
- **Packets:** pydantic schema, deterministic observed text, image findings only above the model's operating
  threshold, regions from anatomical labels (verified on public films).
- **Benchmark (synthetic, 60 stays):** the top driver is a planted unit 93-95 % of the time; IG is the default
  (ExtremalMask 0.85 after tuning on a separate seed, DynaMask 0.21). Half of the selected drivers are unplanted.
- **Stage 3 changed:** trained models are now saved as checkpoints.
- **Not done:** the semi-synthetic benchmark on real tensors and the validation report (lab); your review.

### Stage 3: encoders, fusion model, baselines (build done, 2026-11-16)

- **Code:** `fate_icu.models` (data, image, encoders, fusion, removal, train, arms, run) and
  `fate_icu.eval.predictive`; configs in `configs/models/`.
- **Arms:** B0 (the internship XGBoost) and B1 to B6, M and the Transformer ablation, each with 5
  seeds in the full config; one network class with a presence mask covers every neural arm (D-047).
- **Robustness training:** modality dropout and evidence-unit dropout; what "removed" means as an
  input is defined once in `models/removal.py` for Stages 4 and 7.
- **Metrics:** AUROC, AUPRC, Brier, ECE, MAE, RMSE with patient-level bootstrap CIs on P1 (paired
  test stays) and P2 (all test stays).
- **Found on the way:** the internship's preprocessor turns an all-missing column into 0 without
  a word; the new static encoding stops on it (D-047).
- **Smoke run** (demo, 86 stays, 2 seeds): all 9 arms run end to end on the same test patients.
- **Not done:** lambda tuning, the 5-seed lab run, and bf16 autocast (written, tested on CPU only).

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
- **Real data** (D-042, D-043): the first PubMed pre-flight found 0 records because PubMed
  silently turns `AND NOT` into `AND`; fixed, then 44 searches gave 151,341 records, all
  downloaded and parsed (265,346 passages). StatPearls parsed fully (9,651 articles, 170,780
  passages before dropping the team-outcomes boilerplate). Expected corpus: about 425,000
  passages.

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
