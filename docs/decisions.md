# Design decisions

One entry per decision: date, decision, alternatives considered, reason. Newest last.

---

## D-001 · 2026-10-09 · Billed diagnoses are kept out of every FaTE-ICU input

**Decision.** `Diagnosis_1` and `Diagnosis_2` (ICD-10 chapter of `diagnoses_icd` seq_num 1–2)
stay in the contract with `available_at_t0: false`. They are excluded from `fate_static`,
from all model arms, and from B0 in the comparison grid. Only the legacy scripts
(B0-faithful) use them, to reproduce the internship.
**Alternatives.** Keep them as an input and document the leak; drop them from the contract.
**Reason.** ICD codes are assigned for billing after hospital discharge, so they are not
available at t0 (CLAUDE.md A4). Dropping them entirely would stop B0-faithful matching
the internship. Chosen by the user at Stage 0.

## D-002 · 2026-10-09 · Patient-level hash split 70/10/20, seed 666

**Decision.** `u = sha256("666:<subject_id>")[:8] / 2^64`: train if u < 0.7, val if
u < 0.8, else test. All stays of a patient share a split, and every arm uses the same
split. The internship's stay-level 10 × 80/20 split is recorded for B0-faithful only.
**Alternatives.** Patient hash 80/10/10; 10 repeated grouped splits.
**Reason.** A4 requires patient-level splits, and Stages 3–4 need a validation set.
A hash is identical on every machine and doesn't shift when the cohort changes. The
test share matches the internship's 20 %. Repeated splits would multiply neural-arm
compute about 10× (with 5 seeds each). Chosen by the user.

## D-003 · 2026-10-09 · Long-stay threshold = train-split Q3, then frozen

**Decision.** Threshold = Q3 of `round(los, 2)` over FaTE train stays. The first lab run
prints it (`fate_icu.tabular.freeze_threshold`), and it is then frozen in
`contract.yaml`. Smoke runs compute it from their own train split; full runs refuse to
start until it is frozen.
**Alternatives.** Fixed 4.0 d (the Hempel R code); whole-cohort Q3 (internship, uses test labels).
**Reason.** Same method as the internship, but fit on train only (A4). Chosen by the user.

## D-004 · 2026-10-09 · Narrow exceptions to the `data/` ignore and read-deny rules

**Decision.**
- `.gitignore`:
  - keeps `data/`, which matches any folder of that name;
  - re-includes `src/fate_icu/data/` and `legacy_los/src/data/`, which hold code;
  - adds `legacy_los/results/`.
- `.claude/settings.json`:
  - anchors the data rule to the repository root (`Read(/data/**)`);
  - adds the entries `.gitignore` implied but the settings lacked: `reports/`, `mlruns/`,
    `*.csv`, `*.pth`, `*.ckpt`, `*.gz`, `*.zip`, `legacy_los/results/`, `legacy_los/data/`.

Synthetic fixtures are generated at test time rather than committed as CSV.
**Alternatives.** Rename the package to `src/fate_icu/datasets/`.
**Reason.** Before this change, the legacy loader and `contract.py` would never have been
committed, and the lab clone would have lacked them. Checked: the legacy loader is
readable, and reads under the root `data/` are still denied. Chosen by the user.

## D-005 · 2026-10-09 · The internship's SOFA is frozen as-is; its defects are documented

**Decision.** The static and windowed SOFA features keep the legacy hand-rolled scorer
exactly. Its defects are listed in `docs/contract.md` §7 and not fixed. Stage 2's hourly
SOFA uses the official `mimiciv_derived.sofa`.
**Alternatives.** Fix the scorer (e.g. dobutamine, ventilation for respiratory 3–4) in the adapter.
**Reason.** Fixing it would change a data definition and break B0 reproduction.
Revisit with evidence if the lab results suggest it matters.

## D-006 · 2026-10-09 · Legacy features keep their charttime-only windows

**Decision.** Internship features stay unchanged. They select events by `charttime`
(`starttime` for inputevents) and never check `storetime`. `fate_icu.tabular.audit_leakage`
measures the share of used values stored after t0, on the demo now and in the lab later.
Stage 2's new time series will apply the stricter rule `storetime ≤ t0`.
**Alternatives.** Re-filter legacy inputs by `storetime` in the adapter.
**Reason.** Stage 2 says to leave the internship features unchanged and to record the
difference. A test (`test_late_stored_lab_is_still_used`) documents the gap.
Demo audit (86 stays): 2-3 % of in-window chart, lab and output rows were stored after t0,
median 0.4-0.7 h late; see `docs/contract.md` section 5.

## D-007 · 2026-10-09 · Encoding and scaling fit on train only; unseen categories become missing

**Decision.** The adapter reproduces the legacy encoding (category codes in sorted order,
then min-max to [0, 1], constant columns → 0) but fits it on train stays. A category
not seen in train becomes missing (NaN), which XGBoost handles natively.
**Alternatives.** Fit on the whole cohort, as the internship did.
**Reason.** A4: fit normalisation on train only. Tree models are invariant to these
monotone transforms, so B0's predictions barely change.

## D-008 · 2026-10-09 · `fate_static` = every feature group available at t0

**Decision.** The static input for FaTE-ICU and the B0 arm combine the internship's
groups: admission, demographics, vitals, labs, static SOFA, SOFA trajectory, onset order
and interactions. That is 77 features.
**Alternatives.** Use one of the internship's three feature tables, minus diagnoses.
**Reason.** "The internship features" (Stage 3) is the union. Interactions are built
from static SOFA anyway.

## D-009 · 2026-10-09 · PostgreSQL loader behind the legacy loader interface

**Decision.** `PostgresLoader` implements the legacy `MIMICLoader` methods (`load_table`,
`iter_chunks`) over the schema-qualified tables from the contract. The legacy feature code
runs unchanged on PostgreSQL. An itemid prefilter pushed into SQL only saves time.
**Alternatives.** Port the feature logic to SQL; keep reading CSVs.
**Reason.** No edits to `legacy_los/`, and one code path for the laptop demo database and
the full lab database. Tests show identical tables with and without the prefilter and
from the CSV reader. The `demo` test repeats the comparison on the real demo.

## D-010 · 2026-10-09 · TreeSHAP from XGBoost itself

**Decision.** `explain_shap` uses `Booster.predict(pred_contribs=True)`, which gives
exact TreeSHAP, and sums values by contract feature group.
**Alternatives.** The `shap` package.
**Reason.** Same values with no new dependency. A test checks additivity to the margin.

## D-011 · 2026-10-09 · The legacy snapshot is identified by a content hash

**Decision.** `contract.provenance.legacy_tree_sha256` is a SHA-256 over every file in
`legacy_los/` with CRLF normalised to LF, excluding caches and the `data/`/`results/`
outputs. `test_legacy_snapshot.py` fails if any file changes.
**Alternatives.** None: the internship commit hash is lost.
**Reason.** It guards "never edit legacy_los/" and gives runs a stable identifier in the
manifest.

## D-012 · 2026-10-09 · Reproduction targets are the README's cohort counts

**Decision.** The lab reproduces the cohort flow, whole-cohort Q3 and long-stay share
recorded in `legacy_los/README.md` (`contract.reproduction_targets`). The legacy scripts'
lab run becomes the reference for B0-faithful metrics.
**Alternatives.** None: the report with the original metrics is lost (`docs/internship_metrics.md` is blank).
**Reason.** These are the only internship numbers that survive. The README's partial
metric figures come from incomplete runs (`docs/internship_notes.md`), so they are not used.

## D-013 · 2026-10-09 · Environment pins

**Decision.**
- Python 3.11, installed by uv.
- `pandas` kept below 3 and `numpy` below 2.4: with numpy 2.5 the tests raise about
  77,000 "generic timedelta unit" deprecation warnings inside pandas, which will become
  errors. Below 2.4 there are none.
- `scikit-learn` below 1.10, which removes `LogisticRegression(penalty=None)` (used by the
  legacy classifier).
- `gpu` extra left empty until Stage 6, when bitsandbytes and vLLM get versions compatible
  with the chosen torch.
- mimic-code v2.4.0 on the laptop (demo v2.2) and v3.0.1 in the lab (v3.1).

**Alternatives.** Latest pandas 3; lock vLLM now.
**Reason.** The frozen legacy code was written against pandas 2. Locking
vLLM now would pin torch prematurely. Each mimic-code release matches its MIMIC-IV
version, and laptop numbers are smoke-only.

## D-014 · 2026-10-09 · Contract cohort = legacy cohort rules + legacy preprocessing filters

**Decision.**
1. Legacy `build_cohort` rules.
2. Then the row filters that legacy `prepare_dataset` applies on the `legacy_hempel`
   table: duplicate rows, complete case, vital-mean ranges.

The adapter applies these once, so every feature set shares one cohort.
**Alternatives.** Apply them per feature table, as the legacy scripts do. The result is
identical, because SOFA columns are zero-filled.
**Reason.** One cohort for every arm (A4). Selection uses information from after t0
(LOS window, death in ICU, presence of mappable diagnoses), as in Hempel et al. This is
documented as a limitation, not a feature leak.

## D-015 · 2026-10-09 · B0 stage 2 = XGBoost regressor on predicted-short train stays

**Decision.** As in the internship, the stage-2 regressor is fit on train stays the
classifier predicts short (G-mean threshold chosen on train scores). Its metrics are
reported on predicted-short test stays. `predict()` returns its LOS for every stay;
for stays predicted long, that value is an extrapolation.
**Alternatives.** A second regressor for predicted-long stays.
**Reason.** Faithful to the internship's two-stage design (Hempel Tables 5–6).

## D-016 · 2026-11-02 · Stage 2 variables and item mappings

**Decision.** 30 channels in `configs/variables.yaml`:
- **18 reuse the internship's itemids exactly** (checked against the legacy constants by
  `tests/test_variables.py`).
- **mimic-code mappings for the rest:** systolic and diastolic BP (the `vitalsign` concept's
  items), PaCO2, pH and lactate (the `bg` concept's items), and the derived
  norepinephrine-equivalent and SOFA concepts.
- Temperature uses the °F item only and respiratory rate 220210 only, as the internship
  did. The blood-gas duplicates of chemistry labs (50809, 50811, 50822, 50824) are not used.

**Alternatives.** mimic-code's derived `vitalsign` and `chemistry` tables; adding the °C
temperature and total respiratory rate items.
**Reason.** Stage 2 asks to reuse the internship's mappings. The derived vital-sign and lab
tables drop `storetime`, so the availability rule could not be applied to them.

## D-017 · 2026-11-02 · Window, hourly bins and the availability rule

**Decision.**
- **Window:** values with intime < time ≤ t0, as in the internship's 24 h means (user
  decision). Nothing from before ICU admission is used.
- **Bins:** hour h holds (intime + h, intime + h + 1].
- **Availability:** a raw-table row needs a `storetime`, and it must be ≤ t0. Rows without a
  `storetime` are dropped.
- **Labs** are matched on `subject_id` and time, as Stage 2 specifies. On the demo this
  selects the same rows as the internship's `hadm_id` join.
- **Derived tables have no `storetime`:** SOFA rows are kept when `endtime` ≤ t0, and
  vasopressor intervals are clipped at t0. Their availability cannot be checked, which is a
  documented limitation.

**Alternatives.** Include values exactly at intime; a 6 h lookback before ICU admission;
keep rows without a `storetime`.
**Reason.** You chose the internship's window. Rows whose availability at t0 cannot be shown
are excluded.

Demo smoke build, rows dropped as stored after t0:

| Table | Dropped | Share |
|---|---|---|
| chartevents | 534 of 18,460 | 2.9 % |
| labevents | 41 of 2,364 | 1.7 % |
| outputevents | 38 of 1,346 | 2.8 % |

As a result, heart-rate window means differ from the internship's feature for 55 % of
stays, but only by a median of 0.29 bpm (r = 0.9996).

## D-018 · 2026-11-02 · GCS follows the internship

**Decision.** The GCS total is eye + verbal + motor charted at the same time (all three
required), taking the hourly minimum. The verbal score "No Response-ETT" (intubated) counts
as 1.
**Alternatives.** mimic-code's convention, which scores 15 when the verbal response cannot be
tested and matches the official SOFA CNS score; treating GCS as unmeasured, with an
intubation indicator channel.
**Reason.** User decision, 2026-11-02. Consequence: for intubated patients the GCS channel is
low while the official SOFA CNS channel is 0. In the demo, 41 % of verbal assessments are
ETT. Explanations in Stages 4 and 6 must not read a low GCS in an intubated patient as coma.

## D-019 · 2026-11-02 · Plausibility limits before hourly aggregation

**Decision.**
- Each raw value outside its `valid` limits is dropped: mimic-code's `vitalsign` limits for
  vital signs, generous physiological limits for labs.
- FiO2 values of 1 or less are converted to percent first.
- Clipping to the train percentiles still follows.

**Alternatives.** Clipping only.
**Reason.** Impossible values (in the demo, MAP of −23 and 801, heart rate 0) would distort
hourly means. Clipping moves such a value to the edge of the range instead of removing it.
The demo build dropped 13 chart values.

## D-020 · 2026-11-02 · Urine output subtracts GU irrigant volume in

**Decision.** The internship's 12 urine itemids are used, with 227488 (GU irrigant volume in)
negated, as in mimic-code's `urine_output` concept. A negative hourly net is set to 0.
**Alternatives.** The internship's handling (adding it); mimic-code's derived
`urine_output`, which has no `storetime`.
**Reason.** Irrigant that goes in is not urine, so the internship overstates urine for
irrigated patients (`docs/contract.md` §7, item 6). The floor keeps log1p defined.

## D-021 · 2026-11-02 · Vasopressor channel: zero when no infusion runs

**Decision.**
- The value for each hour is the highest norepinephrine-equivalent dose of any interval
  overlapping it, with intervals clipped at t0, then log1p.
- With no infusion running, the value is 0 with mask 0.
- This channel is never forward-filled and never median-imputed.

**Alternatives.** The general forward-fill-then-median rule.
**Reason.** The infusion record is complete, so "no interval" means no vasopressor. Filling
it forward or with the train median would invent treatment.

## D-022 · 2026-11-02 · SOFA channels are mimic-code's per-hour component scores

**Decision.** The per-hour `respiration`, `coagulation`, `liver`, `cardiovascular`, `cns` and
`renal` columns of `mimiciv_derived.sofa` are used, not the rolling `*_24hours` columns.
NULL means unobserved.
**Alternatives.** The rolling 24 h worst scores, which are never NULL.
**Reason.** A rolling 24 h maximum spreads one event over a whole day, which would distort
the variable × 4-hour attribution in Stage 4. The per-hour scores are sparse (demo:
cardiovascular 93 %, respiratory 5 % of stay-hours) and are forward-filled like the other
channels.

## D-023 · 2026-11-02 · Order of the normalisation steps

**Decision.**
1. Apply log1p where set.
2. Clip to the train 0.5th–99.5th percentiles.
3. Forward-fill within the stay (median channels only).
4. Fill remaining gaps with the train median, with mask 0.
5. Standardise with the train mean and SD of the observed, clipped cells. For the
   vasopressor channel, use all train cells.

`delta` is capped at 24 h. A `raw` array in original units is saved as well, for Stage 4's
"rose from … to …" text. The full config stops if a channel has no train value or is
constant (CLAUDE.md A4); the smoke config only warns.

**Alternatives.** Imputing before clipping; statistics over filled cells.
**Reason.** This follows the Stage 2 spec with train-only statistics (A4). The deterministic
packet strings need the original units.

## D-024 · 2026-11-02 · Reading MIMIC-CXR dates and times

**Decision.**
- The metadata is read with every column as text.
- `StudyDate` must be 8 digits.
- For `StudyTime`, the whole-second part is zero-padded to 6 digits, so `1520.0` means
  00:15:20, and the fraction is kept down to microseconds.
- Invalid values are excluded and counted.
- The file is assumed to have 12 columns: `dicom_id`, `subject_id`, `study_id` and the 10
  documented DICOM fields. The lab run stops if a required column is missing.

**Alternatives.** MedFuse's `int(float(x))`, which drops fractional seconds; reading the
values as numbers.
**Reason.** The PhysioNet page documents neither format. Parsing the text keeps leading
zeros and fractions intact. Dropping fractions would turn images a fraction of a second
apart into ties (the planted case `fraction_breaks_tie`).

## D-025 · 2026-11-02 · Rules for choosing each stay's X-ray

**Decision.**
- Frontal views only: exactly AP or PA.
- Eligible window: intime − 24 h ≤ study time ≤ t0, both ends included; the lookback is
  configurable.
- The most recent eligible image wins. Ties go to AP over PA, then the smallest `dicom_id`.
- Images are matched on `subject_id`. Every stay is kept, with `has_cxr`.
- One image may serve two stays of the same patient; the report counts these.
- The contract's patient split is used, not MIMIC-CXR's.

**Alternatives.** AP only (MedFuse); [intime, outtime] or the first 48 h; MIMIC-CXR's
official split.
**Reason.**
- This follows the Stage 1 spec.
- Bounding at t0 keeps out images taken after the prediction time.
- The result never depends on row order (tested).
- LOS is never used, so pairings are real (tested by shuffling LOS).

## D-026 · 2026-11-02 · Comparison with MedFuse's linkage

MedFuse (`github.com/nyuad-cai/MedFuse`, `datasets/fusion.py`) was read, not copied.

| Aspect | MedFuse | FaTE-ICU |
|---|---|---|
| Views | AP only | AP or PA; AP preferred on ties |
| Window | [intime, outtime]; [intime, intime + 48 h] for mortality | [intime − 24 h, t0] |
| Choice | latest; ties depend on row order | latest; ties AP, then `dicom_id` |
| StudyTime | `int(float(x))` zero-padded; fractions dropped | fractions kept |
| Stays without an image | dropped from the paired set | kept, `has_cxr` = false |
| Split | their own, derived from EHR list files | the contract's patient split |

**Reason.** MedFuse's whole-stay window would let in images taken after t0 for a
length-of-stay task. Its tie handling is not deterministic. Their `create_split.py` probably
compares subject IDs as strings against integers; that does not affect us.

## D-027 · 2026-11-02 · X-ray preprocessing and cache

**Decision.**
1. Read with Pillow to 2-D grayscale. Colour is averaged per pixel, as the torchxrayvision
   README does; alpha is dropped; the maximum is 255 or 65535 by bit depth; other modes are
   an error.
2. `xrv.datasets.normalize`.
3. `XRayCenterCrop`.
4. `XRayResizer(224)` as float32 for the model, and `XRayResizer(512)` stored as uint8 for
   display.

The cache holds one compressed `.npz` per `dicom_id` in `artifacts/cxr_cache/`. It
resumes, and writes are atomic. Missing or unreadable images are counted, their keys go only
to a local `problems.txt`, and the command exits with an error.

torchxrayvision is imported inside `warnings.catch_warnings()`. Version 1.5.7 adds a
global "ignore all warnings" filter when imported (verified, and guarded by a test). The
output matches the README pipeline on the public images within 0.05 (float rounding).

**Alternatives.** `xrv.utils.load_image` (takes the first channel, not the mean); the cv2
resize engine; a float16 cache.
**Reason.** The same conventions as the data the torchxrayvision encoder was trained on, with
no extra dependencies.

## D-028 · 2026-11-02 · Public test images

**Decision.** Four images in `tests/fixtures/cxr/`, credited in `SOURCES.md`:
- two NIH ChestX-ray14 PNGs from torchxrayvision's repository (the licence wording could not
  be verified; included at your request);
- a CC0 PA PNG from Wikimedia Commons;
- a CC BY-SA 4.0 bedside AP JPEG from Wikimedia Commons (attribution and share-alike apply
  to that image).

**Alternatives.** Synthetic images only.
**Reason.** Stage 1 asks for preprocessing tested on real public X-rays. Synthetic images
still cover the edge cases: colour, 16-bit, non-square, and corrupt files.

## D-029 · 2026-11-02 · Stage 1 dependencies

**Decision.**
- torch 2.15.0. From PyPI it is CPU-only on Windows. On Linux the wheel pulls CUDA 13
  libraries, so the lab's NVIDIA driver must support CUDA 13 (R580 or later; check with
  `nvidia-smi` before Stage 3).
- torchvision 0.29.2, torchxrayvision 1.5.7, scikit-image 0.26.0, pillow 13.0.0.
- Ranges: torch below 3 (A6: PyTorch 2.x); torchxrayvision 1.5.x.

**Alternatives.** Re-implement the preprocessing without torchxrayvision.
**Reason.** Stage 1 specifies torchxrayvision's normalisation, and Stage 3 needs it for the
encoder anyway.

## D-030 · 2026-11-02 · A date-alignment check in the flow report

**Decision.** The report gives the share of cohort patients with any dated X-ray who have one
between intime − lookback and the outtime of one of their ICU stays.
**Alternatives.** Trust the Stage 1 background.
**Reason.** The PhysioNet page does not confirm that MIMIC-CXR uses MIMIC-IV's per-patient
date shift. If it did not, this share would be near zero and the linkage meaningless.

## D-031 · 2026-11-13 · Stage 5 dependencies and the MedCPT revisions

**Decision.**
- bm25s 0.3.16 with PyStemmer 3.1.0, faiss-cpu 1.15.1, transformers 5.22.2 (tokenizers
  0.23.3), pypdf 7.0.0, and huggingface-hub 2.4.0. huggingface-hub is declared because the
  code calls it; transformers already required it.
- MedCPT at fixed commits: query encoder `d83a36cc…`, article encoder `d05a736d…`,
  cross-encoder `71caf65d…` (about 1.3 GB in the Hugging Face cache, downloaded 2026-11-10).
- Checked offline under transformers 5.22.2: all three reproduce the outputs printed on their
  model cards (article embeddings to 4 decimals, cross-encoder logits within 0.0001).
- MedCPT is a United States Government work (public domain, per its LICENSE file).

**Alternatives.** rank-bm25; sentence-transformers.
**Reason.** The Stage 5 spec names bm25s, MedCPT and FAISS. These versions are newer than most
examples online, so the code follows the installed APIs:
- transformers 5's `BertTokenizer` takes a vocabulary dict and ignores `vocab_file=`;
- bm25s fails on an empty query;
- FAISS returns tied scores in arbitrary order and pads results past the index size with -1.

## D-032 · 2026-11-13 · Where the Stage 5 data lives

**Decision.** `FATE_RAG_DIR` holds everything; its default is `<FATE_ARTIFACTS_DIR>/rag`.
- `raw/` holds the downloads, shared by all configs.
- Each config has its own build folder (`rag_smoke/`, `rag_full/`) with `corpus/<id>/`,
  `index/<version>/`, `eval/`, and the pointers `corpus.CURRENT` and `index.CURRENT`.
- The full config refuses a `FATE_RAG_DIR` inside OneDrive.
- `NCBI_EMAIL` and `NCBI_API_KEY` are secret settings: sent only to NCBI, never printed,
  never written to a manifest.

**Alternatives.** `artifacts/` inside the repository.
**Reason.** The data is several GB and the repository is in OneDrive. A smoke run must never
replace the full index.

## D-033 · 2026-11-13 · PubMed through NCBI E-utilities

**Decision (your choice 1A).**
- `configs/rag/pubmed_queries.yaml` holds 7 topic searches (the Stage 5 topics) and 37 driver
  searches that cover all 50 driver keys.
  - Every search is ANDed with an ICU block: MeSH Intensive Care Units, Critical Care and
    Critical Illness, plus title/abstract terms.
  - Driver searches are also ANDed with an outcome block: length of stay, mortality,
    prognosis or outcome.
  - Filters: English, with an abstract, not animal-only, not paediatric-only, no comments,
    editorials, letters, news, errata, retractions or expressions of concern; 1990 to 2026.
- Each search is sliced by publication date (the whole range, then years, months, days), so
  no slice reaches esearch's 10,000-id cap. The slices must return the search's full count,
  or the fetch stops.
- The PMIDs are saved in `raw/pubmed/pmids.json`, which pins the corpus.
- Records are fetched in resumable batches of 200, at most 3 requests a second (10 with an
  API key). The fetch stops if more than 1% of records are missing.
- Parsing drops records without a title or abstract, non-English records, retracted
  articles (by publication type or retraction notice) and non-research publication types.
  Structured-abstract labels stay inline, and the section title is empty, so MedCPT sees
  (title, abstract) as it was trained.

**Alternatives.** MedRAG's 24-million-snippet copy, filtered by keywords on Kaggle.
**Reason.** MeSH-based selection, current literature, runs on the laptop, and the saved PMID
list pins the corpus. The search terms are Claude's draft, to be reviewed before the full
fetch.

## D-034 · 2026-11-13 · All of StatPearls

**Decision (your choice 3).**
- The NCBI Bookshelf archive `statpearls_NBK430685.tar.gz`: 2,047,364,463 bytes,
  Last-Modified 2026-11-08, pinned as `expected_size` in the full config. The download
  resumes after an interruption and records a SHA-256 sidecar.
- The archive is streamed, never extracted. Only `.nxml` members are read. Unsafe or
  oversized members stop the build, and so do more than 1% unparseable articles.
- Dropped sections: "Continuing Education Activity", "Review Questions", "References",
  "Disclosure(s)" and, added after the real-archive check, "Enhancing Healthcare Team
  Outcomes" (D-043).
- Nested sections become "Parent: Child". Citation markers, figures, tables, captions and
  reference lists are removed.
- Licence CC BY-NC-ND 4.0: passages are quoted with attribution, and the parsed copy is
  never published.

**Alternatives.** An ICU-relevant subset; MedRAG's build script.
**Reason.**
- There is no filter to justify.
- General articles (anaemia, hyperkalaemia, lactic acidosis) explain our channels.
- The re-ranker pushes off-topic passages down.

The parser was built on synthetic files and then checked on the real archive (D-043).

## D-035 · 2026-11-13 · Guidelines

**Decision (your choice 4).**
- Sources: SSC 2021, KDIGO AKI 2012, ESICM ARDS 2023 and PADIS 2018, as PDFs you download
  into `raw/guidelines/` under the file names in `configs/rag/full.yaml`. The build stops
  while a licence still says "TO CONFIRM".
- PDF text comes from pypdf's plain mode, which keeps two-column pages in reading order.
- Running headers, footers and page numbers are dropped, and hyphenated line breaks are
  rejoined.
- Headings are lines that are numbered, start with a keyword (Recommendation, Chapter,
  Section, Appendix, Part, Annex), or are in Title Case or capitals.
- Excluded sections are skipped: references, acknowledgements, funding, conflicts of
  interest and similar. Inside a reference list only a Chapter, Section, Appendix, Part or
  Annex heading ends the skip.
- Page labels are kept as locators ("p. 1183").
- JATS XML (e.g. from PubMed Central) is accepted too.

**Alternatives.** Heading detection by font size; plain text without sections.
**Reason.** Sections and page locators make citations checkable. Known limit: unnumbered
sentence-case headings are not detected, so their text stays in the previous section. The
sections of the four real PDFs will be inspected once they are downloaded, and page ranges
set if needed.

## D-036 · 2026-11-13 · Passages

**Decision.**
- A passage holds at most 300 tokens of the MedCPT article tokenizer. It is split at
  sentence boundaries and never crosses a section.
- Within a section the split uses the fewest chunks, then the most even ones, so chunks hold
  150 to 300 tokens unless the section is short. Passages under 20 tokens are dropped (5 at
  first; D-043).
- Text is NFKC-normalised. Each passage keeps its title, section and page locator, and its
  id is `<source>:<doc_id>:<chunk>`.
- The corpus is a Parquet file sorted by id, hashed over its canonical JSON content.

**Alternatives.** Whole abstracts (MedRAG); fixed word windows; a sentence-splitter package
(another dependency).
**Reason.** 300 tokens fit both the cross-encoder and the Stage 7 NLI checker (512 tokens
together with the claim). Balanced chunks avoid tiny tail passages. Content hashing gives the
same corpus id on every machine.

## D-037 · 2026-11-13 · Retrieval

**Decision.**
- **BM25:** the Lucene variant, k1 1.5, b 0.75, English stopwords and the Snowball stemmer,
  over title, section and text. Only positive scores count as retrieved.
- **Dense:** the MedCPT article encoder over (title, "section. text"), the CLS vector and the
  raw inner product, in an exact FAISS `IndexFlatIP`, fp32.
- **Hybrid:** reciprocal-rank fusion (k = 60) of the two top-50 lists.
- **Rerank:** the MedCPT cross-encoder on "Title. text" for the first 50 fused candidates.
- **Result:** the top 3 per driver, deduplicated, numbered R1…Rn in order of first appearance.
- **Ties:** every tie is broken by corpus row (the passage id). FAISS itself returned tied
  results in the order [3, 2, 1, 0]; when a tie could straddle the cut-off, that query is
  scored exactly against all vectors.

**Alternatives.** Approximate FAISS indexes (HNSW, IVF); cosine similarity; reranking every
fused candidate.
**Reason.** This follows the Stage 5 spec and MedCPT's own usage, and exact search keeps
retrieval deterministic.

## D-038 · 2026-11-13 · Queries, the query bank and the packet key

**Decision.**
- Template: "{label} associated with {longer|shorter} intensive care unit length of stay".
- `configs/rag/queries.yaml` holds a label for each of the 50 driver keys. A label that
  differs from its source says why, for example:
  - "hemoglobin" (US spelling);
  - "SOFA score" without "(hourly)";
  - static groups written out without numbers;
  - image concepts phrased "… on chest radiograph".
- A label may hold no digits outside words, no percentages and no units (tested).
- All 100 queries are run once into `query_bank.json`. `retrieve(packet)` looks up
  (modality, key, direction), so it is deterministic and needs no model at explanation time.
- **Stage 4 packets must give each driver its canonical `key`** (channel name, static group
  or image concept). Retrieval never reads a driver's label, observed, time or effect fields.

**Alternatives.** Queries built from packet labels; live retrieval for each packet.
**Reason.** Packet labels can carry patient values (Stage 4's example image label contains a
probability). A fixed bank makes the Stage 6 data reproducible across machines.

## D-039 · 2026-11-13 · The GPU step

**Decision.**
- One command, `fate_icu.rag.gpu_step`, does the model-heavy work: it embeds the passages
  (resumable shards), builds FAISS, encodes the queries and reranks.
  - It runs on a Kaggle or Colab GPU for the full corpus, and on the laptop CPU for the
    smoke corpus, with the same code.
  - It reads a bundle the laptop prepares: a fresh wheel of the code, the corpus,
    `gpu_spec.json`, and the laptop's BM25 lists.
- Back on the laptop, `build_index --part dense` checks the output before installing it:
  - the embeddings' corpus hash and model;
  - the BM25 lists, which must be identical;
  - the dense lists, recomputed for every query (mean overlap at least 0.9);
  - the reranking, recomputed for 2 queries.

**Alternatives.** Reranking on the laptop.
**Reason.** Measured on this i7-2600 CPU:
- the cross-encoder scores 1.5 pairs a second, so the full bank's 5,000 rerank pairs would
  take about an hour;
- the article encoder embeds 1.6 passages a second, so the corpus would take about 90 hours.

This is the split in the approved plan.

## D-040 · 2026-11-13 · Retrieval evaluation protocol

**Decision.**
- **Queries:** the 50 "longer" queries.
- **Pool:** the union of each system's top 5, shown in a blind, deterministic order.
- **Grades:** 0, 1 or 2 (docs/rag_grading_guide.md).
- **Metrics:**
  - P@3 with relevant meaning grade 1 or more (primary), and P@3 at grade 2 (strict);
  - nDCG@5 with gains 2^g − 1 and the ideal ordering of all judged passages of the query.
- **Confidence intervals:** percentile bootstrap, 95%, over 1,000 resamples of queries.
- **Comparisons:** hybrid vs BM25, hybrid vs dense, hybrid + rerank vs hybrid, and dense vs
  BM25. Each has a paired bootstrap CI and a two-sided sign-flip permutation p (10,000
  permutations, exact when possible), with Holm correction within each metric.
- Labels from another index version are refused.
- `configs/rag/relevance_labels.tsv` stores ids and grades only.

**Alternatives.** Binary relevance; relevance judged by an LLM.
**Reason.** Graded labels serve both precision and nDCG, and human labels are the reference
standard. The labels file can be committed because it holds no copyrighted text.

## D-041 · 2026-11-13 · Index versions and the lab copy

**Decision.**
- The index version is the first 16 hex characters of a SHA-256 over:
  - the corpus hash;
  - the BM25 parameters;
  - both MedCPT encoders' revisions;
  - the embedding settings.
- `index.json` records the SHA-256 of every file and the query-bank hash, and downstream run
  manifests record both.
- `fate_icu.rag.verify` re-checks a copied index in the lab.

**Reason.** The Stage 5 spec: "Version the index (corpus hash + encoder revision) in the run
manifest."

## D-042 · 2026-11-13 · PubMed pre-flight: a silent query bug, and what the searches return

**Decision.**
- `fate_icu.rag.fetch --source pubmed --ids-only` runs every search without downloading any
  record. It reports the records found per search and PubMed's own notes (a phrase or field it
  did not recognise). A search that finds nothing stops the run, with PubMed's reading of it.
- The first pre-flight found 0 records for all 44 searches. PubMed has no `AND NOT`: it
  rewrites "A AND NOT B" into "A AND B" without an error (checked: `sepsis[tiab] AND NOT
  editorial[pt]` found the 2,033 sepsis editorials, `sepsis[tiab] NOT editorial[pt]` the other
  151,258). The filter had therefore selected editorials and letters instead of excluding
  them.
- Fixed: exclusions are `NOT (...)`; "abstract" is `hasabstract[text]`; the phrases
  `"retraction of publication"[pt]` (not a publication type in PubMed's index) and
  `"type of admission"[tiab]` (not found) are removed. A test rejects "AND NOT" and "OR NOT" in
  any search.
- Corrected pre-flight: 44 searches, **151,341 unique PMIDs**, 12 minutes, no PubMed notes. Per
  search 418 to 41,827 records. The 7 topic searches find 7,946 to 41,827 each.
- The record download is incremental. Batch files are named by their content, any readable
  batch counts whichever search list fetched it, and a PMID PubMed returns nothing for is
  remembered. Changing a search therefore downloads only the new records.
- Book chapters (`PubmedBookArticle`, which includes StatPearls chapters) are read and excluded
  as `book_article`. They count as found, so they cannot trip the 1% missing-records check, and
  they do not duplicate the StatPearls archive.

**Alternatives.** Trust the searches without a pre-flight.
**Reason.** A query that matches nothing is not an error in PubMed. Without the pre-flight the
build would have run on an empty corpus, or on a corpus of editorials.

**Result on the real records (2026-11-13).**
- All 151,341 records were downloaded in 19 minutes (757 batches, 2.4 GB), none missing.
- Dry-run parse (8 minutes): 151,140 kept, 190 book chapters excluded, 11 without an
  abstract. Nothing was left to the language, retraction or publication-type rules, which the
  search filter already applies.
- 151,126 abstracts gave 265,346 passages: 45,226 abstracts in one passage, 98,733 in two,
  6,476 in three. Median 198 tokens, 95th percentile 281, maximum 300. 0.15% exact duplicates.
- Expected full corpus: about 425,000 passages (PubMed 265,000, StatPearls about 160,000,
  guidelines a few hundred), about 1-1.5 hours of embedding on a Kaggle T4.

## D-043 · 2026-11-13 · StatPearls on the real archive

**Decision.**
- The archive (2,047,364,463 bytes, SHA-256 `08117bcf…`) was downloaded on 2026-11-13 in 8.8
  minutes. It holds 9,653 articles: 9,651 parsed, one without a title and one without a body.
- Real articles are `article-<number>` files with no NBK id. Their URL is
  `https://www.ncbi.nlm.nih.gov/books/n/statpearls/article-<number>/`, Bookshelf's address for
  that part. NCBI answers scripts with a bot check, so the pattern is not verified by the build:
  open a few URLs from `corpus_manifest.csv` in a browser. The 117 `nurse-article-<number>`
  articles (1.2% of passages) link to the whole book.
- "Enhancing Healthcare Team Outcomes" is also dropped. It is process text about
  interprofessional teams: 83% of its passages use team language and 57% mention "outcomes"
  (9% elsewhere), and the 9,072 passages (5.3% of StatPearls) support no driver. Every query
  ends in "length of stay", so they would keep competing with real evidence. This narrows "all
  of StatPearls" by one section type; re-adding it is one line in
  `sources.statpearls.exclude_sections`.
- `chunking.min_tokens` is 20, not 5. Passages under 20 tokens (0.6%) are list fragments such as
  "Technician / Physiatrist".
- Dry run of the chunker on the whole archive (before those two changes): 170,780 passages in
  327 s, median 215 tokens, 95th percentile 288, maximum 300; 0.24% exact duplicate texts.

**Alternatives.** Keep every section; keep short passages.
**Reason.** Checked on the real data rather than assumed. Both choices are cheap to reverse
until the index and the relevance grades exist, and expensive afterwards.

## D-044 · 2026-11-15 · Moving the project between computers

**Decision.**
- The code goes through a private git remote; a `git bundle` is the backup copy.
- The Stage 5 downloads are copied, not fetched again, because PubMed changes daily and NCBI
  replaces the StatPearls archive (the corpus is pinned by `pmids.json` and the archive size).
- Folders are copied together with a `SHA256SUMS.txt` (`fate_icu.utils.checksums`, the format of
  `sha256sum`, hashes of raw bytes) and checked on arrival. `binary_sha256` moved to
  `utils/hashing.py` for this; `rag/version.py` still exports it.
- The demo database can be restored from a `pg_dump` (PostgreSQL 18 or newer) or rebuilt from
  `C:\mimic` (any version from 14).
- Steps are in `docs/laptop_setup.md`, section 6.

**Alternatives.** Zip the repository; re-download the data.
**Reason.** A zip hides damage and drops history; re-downloading would change the corpus.
