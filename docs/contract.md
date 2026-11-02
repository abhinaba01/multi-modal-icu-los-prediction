# FaTE-ICU contract

The internship's definitions, frozen. The machine-readable version is
`configs/contract.yaml`, loaded through `fate_icu.data.contract.load_contract()`; later
code imports it instead of hard-coding values. This document gives the evidence behind
each entry. Changing a definition means asking first and recording it in
`docs/decisions.md`.

Audited at Stage 0 (2026-10-09) from the code in `legacy_los/`.

## 1. Provenance

| Item | Value |
|---|---|
| Internship code | `legacy_los/`, tag `v1.0-internship`, a read-only snapshot |
| Internship commit hash | **unavailable**: the original repository is gone |
| Original test metrics | **unavailable**: the project report is lost; `docs/internship_metrics.md` is intentionally blank |
| Snapshot identity | `legacy_tree_sha256 = 3abe167bbf7cfe970acd0fb8af2d62d8c5883cc558bfbbf3e19d260e75f63a6a` (LF-normalised hash of `legacy_los/`; `tests/test_legacy_snapshot.py` fails if any file changes) |
| What the internship replicates | Hempel, Sadeghi, Kirsten (2023), *Prediction of ICU Length of Stay in the MIMIC-IV Dataset*, Appl. Sci. 13:6930, plus static and dynamic SOFA extensions |
| Reproduction targets | Cohort counts from `legacy_los/README.md` (§10). The lab run of the legacy scripts becomes the metric reference |

## 2. Data source

- **MIMIC-IV version:** v3.1 (internship and lab). The paper used v2.1, so the internship
  cohort is larger; that is the one documented deviation from the paper. The laptop uses
  the open **MIMIC-IV Clinical Database Demo v2.2**.
- **How the internship read it:** pandas streaming the raw CSV(.gz) files
  (`legacy_los/src/data/loader.py`). `legacy_los/sql/Extraction.sql` is an annotated copy
  of the Hempel GitLab SQL kept as documentation; it is not executed and uses unqualified
  table names.
- **FaTE-ICU:** the same legacy feature code, fed from PostgreSQL by
  `fate_icu.tabular.loaders.PostgresLoader` (decision D-009). Tests show identical
  results to the CSV reader.
- **Schema names:** `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`, as created by
  mimic-code (v3.0.1 in the lab, v2.4.0 for the demo).
- **External asset:** the public CMS 2018 ICD-9 → ICD-10 GEM file (`2018_I9gem.txt`),
  used by the diagnosis features.

## 3. Cohort

Unit: ICU stay (`mimiciv_icu.icustays`). Rules in order, from `legacy_los/src/features/cohort.py`
and `src/preprocessing/prepare.py`:

| # | Rule | Source |
|---|---|---|
| 1 | All ICU stays; **not** first-stay-only (a patient may contribute several stays) | `cohort.build_cohort` |
| 2 | Exclude ICU readmissions < 48 h after the previous ICU discharge **in the same `hadm_id`**. The previous stay comes from all stays, excluded or not; the first stay of an admission is always kept | `_non_readmission_stays` |
| 3 | Exclude death in the ICU: `admissions.deathtime ≤ icustays.outtime` | `build_cohort` |
| 4 | Keep `1 < los ≤ 21` days | `build_cohort` |
| 5 | Keep `anchor_age ≥ 18` (no-op on MIMIC-IV) | `build_cohort` |
| 6 | Drop duplicate feature rows | `prepare_dataset` step 1 |
| 7 | Complete case on the `legacy_hempel` features (including both diagnosis chapters) | step 2 |
| 8 | Drop stays whose per-stay **mean** vital is out of range: heart rate [25, 225], respiration (7, 40), SpO2 (50, 120), temperature [86, 113] °F | step 3 |

- **Stays that end before t0:** excluded by construction, since rule 4 requires
  LOS > 1 day = W.
- **Selection uses information from after t0:** LOS and outtime (rule 4), death time
  (rule 3), and whether both billed diagnoses map to a chapter (rule 7). This is inherent
  to the Hempel design. It limits prospective use, but these values are never model inputs.
- **Cohort bias:** the complete-case rule keeps only stays with all 7 vitals and all
  18 labs in the first 24 h.

## 4. Target

- **Total ICU LOS** (`mimiciv_icu.icustays.los`, fractional days), not remaining LOS.
  The regression target is `round(los, 2)`.
- **Long stay:** `round(los, 2) > threshold`.
  - Internship: threshold = Q3 over the whole final cohort, test included (4.11 d on v3.1).
  - FaTE-ICU (D-003): Q3 over the FaTE train split, frozen in `contract.yaml` after the
    first lab run. It is `null` until then.

## 5. Window and timestamps

- **W = 24 h; t0 = `icustays.intime` + 24 h.**
- **Vital and lab means:** events in `(intime, t0]`. An event exactly at intime is
  excluded; one exactly at t0 is included.
- **SOFA:** events in `[intime, t0)`, worst value, also split into four 6-hour windows.
- **Timestamps:** `charttime` (chartevents, labevents, outputevents) and `starttime`
  (inputevents). **`storetime` is never checked.** A lab drawn before t0 counts even if
  its result posted after t0. `fate_icu.tabular.audit_leakage` measures how often
  (D-006); Stage 2's new time series require `storetime ≤ t0`.
  **Demo measurement (86 stays, 2026-11-02):** of the rows the legacy windows select,
  2.8 % of chartevents, 2.1 % of labevents and 2.8 % of outputevents were stored after
  t0, typically 0.4–0.7 h after. 69 % of stays have at least one such chart row. The
  effect on feature values is small (a few late values among many in a 24 h mean) but
  not zero. The full-data figure comes from the lab run (runbook 0.6).
- Labs have no `stay_id`. They are joined on `hadm_id` and windowed by each stay's own intime.

## 6. Features

79 features in 9 groups. `fate_static` (the FaTE-ICU static input and the B0 arm) is
every group available at t0: 77 features. The internship's three tables are
`legacy_hempel` (32), `legacy_hempel_sofa` (41) and `legacy_hempel_sofa_dynamics` (70).

| Group | n | Window | Available at t0 |
|---|---|---|---|
| admission details | 3 | known at ICU admission | yes |
| billed diagnoses | 2 | assigned after hospital discharge | **no** |
| demographics | 2 | known at ICU admission | yes |
| vital signs (mean of first 24 h) | 7 | (intime, t0] | yes |
| laboratory results (mean of first 24 h) | 18 | (intime, t0] | yes |
| SOFA score (worst of first 24 h) | 9 | [intime, t0) | yes |
| SOFA trajectory across 6-hour windows | 18 | [intime, t0) in 4 × 6 h | yes |
| order of organ-dysfunction onset | 14 | [intime, t0) in 4 × 6 h | yes |
| SOFA organ interaction terms | 6 | [intime, t0) | yes |

**Time-window check.** Every feature's window ends at or before t0 except the two billed
diagnoses, which are coded after discharge. The contract model refuses any feature in a
group marked "available at t0" whose window ends after t0. `tests/test_no_leakage.py`
checks the actual data read: the latest event behind each stay's features is never after
t0, and extreme events added after every t0 change no feature.

Per feature (generated from `configs/contract.yaml`):

| # | Feature | Plain-English label | Group | Source | Window | ≤ t0 |
|---|---|---|---|---|---|---|
| 1 | `first_careunit` | first ICU care unit | admission | mimiciv_icu.icustays.first_careunit | at/before ICU admission | yes |
| 2 | `admission_type` | hospital admission type | admission | mimiciv_hosp.admissions.admission_type | at/before ICU admission | yes |
| 3 | `admission_location` | where the patient was admitted from | admission | mimiciv_hosp.admissions.admission_location | at/before ICU admission | yes |
| 4 | `Diagnosis_1` | ICD-10 chapter of the first billed diagnosis | diagnoses | mimiciv_hosp.diagnoses_icd seq_num 1 (ICD-9 mapped with the CMS 2018 GEM) | after hospital discharge | **no** |
| 5 | `Diagnosis_2` | ICD-10 chapter of the second billed diagnosis | diagnoses | mimiciv_hosp.diagnoses_icd seq_num 2 (ICD-9 mapped with the CMS 2018 GEM) | after hospital discharge | **no** |
| 6 | `gender` | sex | demographics | mimiciv_hosp.patients.gender | at/before ICU admission | yes |
| 7 | `anchor_age` | age (years) | demographics | mimiciv_hosp.patients.anchor_age | at/before ICU admission | yes |
| 8 | `heart_rate` | mean heart rate (beats/min) | vitals | mimiciv_icu.chartevents itemid 220045 | (intime, t0] | yes |
| 9 | `respiration` | mean respiratory rate (breaths/min) | vitals | mimiciv_icu.chartevents itemid 220210 | (intime, t0] | yes |
| 10 | `saturation` | mean oxygen saturation, SpO2 (%) | vitals | mimiciv_icu.chartevents itemid 220277 | (intime, t0] | yes |
| 11 | `temperature` | mean body temperature (degrees F) | vitals | mimiciv_icu.chartevents itemid 223761 | (intime, t0] | yes |
| 12 | `gcs_eye` | mean GCS eye-opening score | vitals | mimiciv_icu.chartevents itemid 220739 | (intime, t0] | yes |
| 13 | `gcs_verbal` | mean GCS verbal score | vitals | mimiciv_icu.chartevents itemid 223900 | (intime, t0] | yes |
| 14 | `gcs_motor` | mean GCS motor score | vitals | mimiciv_icu.chartevents itemid 223901 | (intime, t0] | yes |
| 15 | `anion_gap` | mean anion gap (mEq/L) | labs | mimiciv_hosp.labevents itemid 50868 | (intime, t0] | yes |
| 16 | `bicarbonate` | mean serum bicarbonate (mEq/L) | labs | mimiciv_hosp.labevents itemid 50882 | (intime, t0] | yes |
| 17 | `chloride` | mean serum chloride (mEq/L) | labs | mimiciv_hosp.labevents itemid 50902 | (intime, t0] | yes |
| 18 | `creatinine` | mean serum creatinine (mg/dL) | labs | mimiciv_hosp.labevents itemid 50912 | (intime, t0] | yes |
| 19 | `glucose` | mean blood glucose (mg/dL) | labs | mimiciv_hosp.labevents itemid 50931 | (intime, t0] | yes |
| 20 | `magnesium` | mean serum magnesium (mg/dL) | labs | mimiciv_hosp.labevents itemid 50960 | (intime, t0] | yes |
| 21 | `potassium` | mean serum potassium (mEq/L) | labs | mimiciv_hosp.labevents itemid 50971 | (intime, t0] | yes |
| 22 | `sodium` | mean serum sodium (mEq/L) | labs | mimiciv_hosp.labevents itemid 50983 | (intime, t0] | yes |
| 23 | `urea_nitrogen` | mean blood urea nitrogen (mg/dL) | labs | mimiciv_hosp.labevents itemid 51006 | (intime, t0] | yes |
| 24 | `hematocrit` | mean haematocrit (%) | labs | mimiciv_hosp.labevents itemid 51221 | (intime, t0] | yes |
| 25 | `hemoglobin` | mean haemoglobin (g/dL) | labs | mimiciv_hosp.labevents itemid 51222 | (intime, t0] | yes |
| 26 | `mch` | mean corpuscular haemoglobin, MCH (pg) | labs | mimiciv_hosp.labevents itemid 51248 | (intime, t0] | yes |
| 27 | `mchc` | mean corpuscular haemoglobin concentration, MCHC (g/dL) | labs | mimiciv_hosp.labevents itemid 51249 | (intime, t0] | yes |
| 28 | `mcv` | mean corpuscular volume, MCV (fL) | labs | mimiciv_hosp.labevents itemid 51250 | (intime, t0] | yes |
| 29 | `platelet_count` | mean platelet count (K/uL) | labs | mimiciv_hosp.labevents itemid 51265 | (intime, t0] | yes |
| 30 | `rdw` | mean red cell distribution width, RDW (%) | labs | mimiciv_hosp.labevents itemid 51277 | (intime, t0] | yes |
| 31 | `red_blood_cells` | mean red blood cell count (m/uL) | labs | mimiciv_hosp.labevents itemid 51279 | (intime, t0] | yes |
| 32 | `white_blood_cells` | mean white blood cell count (K/uL) | labs | mimiciv_hosp.labevents itemid 51301 | (intime, t0] | yes |
| 33 | `sofa_resp` | respiratory SOFA sub-score (worst, first 24 h) | sofa_static | PaO2 50821 / FiO2 223835 (SpO2 220277 fallback) | [intime, t0) | yes |
| 34 | `sofa_cardio` | cardiovascular SOFA sub-score (worst, first 24 h) | sofa_static | inputevents vasopressor rates; MAP 220052/220181/225312 | [intime, t0) | yes |
| 35 | `sofa_hepatic` | liver SOFA sub-score (worst, first 24 h) | sofa_static | labevents bilirubin 50885 | [intime, t0) | yes |
| 36 | `sofa_coag` | coagulation SOFA sub-score (worst, first 24 h) | sofa_static | labevents platelets 51265 | [intime, t0) | yes |
| 37 | `sofa_renal` | renal SOFA sub-score (worst, first 24 h) | sofa_static | labevents creatinine 50912; outputevents urine | [intime, t0) | yes |
| 38 | `sofa_neuro` | neurological (GCS) SOFA sub-score (worst, first 24 h) | sofa_static | chartevents GCS 220739+223900+223901 | [intime, t0) | yes |
| 39 | `sofa_total` | total SOFA score (worst, first 24 h) | sofa_static | the six sub-scores above | [intime, t0) | yes |
| 40 | `urine_24h` | total urine output in first 24 h (mL) | sofa_static | mimiciv_icu.outputevents urine itemids | [intime, t0) | yes |
| 41 | `bilirubin_max` | highest total bilirubin in first 24 h (mg/dL) | sofa_static | mimiciv_hosp.labevents itemid 50885 | [intime, t0) | yes |
| 42 | `sofa_resp_delta` | change in respiratory SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly respiratory SOFA | [intime, t0), 4 × 6 h | yes |
| 43 | `sofa_resp_slope` | trend of respiratory SOFA over the first 24 h | sofa_trajectory | 6-hourly respiratory SOFA | [intime, t0), 4 × 6 h | yes |
| 44 | `sofa_resp_onset_win` | first 6-hour window with respiratory dysfunction | sofa_onset | 6-hourly respiratory SOFA | [intime, t0), 4 × 6 h | yes |
| 45 | `sofa_cardio_delta` | change in cardiovascular SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly cardiovascular SOFA | [intime, t0), 4 × 6 h | yes |
| 46 | `sofa_cardio_slope` | trend of cardiovascular SOFA over the first 24 h | sofa_trajectory | 6-hourly cardiovascular SOFA | [intime, t0), 4 × 6 h | yes |
| 47 | `sofa_cardio_onset_win` | first 6-hour window with cardiovascular dysfunction | sofa_onset | 6-hourly cardiovascular SOFA | [intime, t0), 4 × 6 h | yes |
| 48 | `sofa_hepatic_delta` | change in liver SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly liver SOFA | [intime, t0), 4 × 6 h | yes |
| 49 | `sofa_hepatic_slope` | trend of liver SOFA over the first 24 h | sofa_trajectory | 6-hourly liver SOFA | [intime, t0), 4 × 6 h | yes |
| 50 | `sofa_hepatic_onset_win` | first 6-hour window with liver dysfunction | sofa_onset | 6-hourly liver SOFA | [intime, t0), 4 × 6 h | yes |
| 51 | `sofa_coag_delta` | change in coagulation SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly coagulation SOFA | [intime, t0), 4 × 6 h | yes |
| 52 | `sofa_coag_slope` | trend of coagulation SOFA over the first 24 h | sofa_trajectory | 6-hourly coagulation SOFA | [intime, t0), 4 × 6 h | yes |
| 53 | `sofa_coag_onset_win` | first 6-hour window with coagulation dysfunction | sofa_onset | 6-hourly coagulation SOFA | [intime, t0), 4 × 6 h | yes |
| 54 | `sofa_renal_delta` | change in renal SOFA (creatinine), hours 0-6 to 18-24 | sofa_trajectory | 6-hourly renal SOFA (creatinine only) | [intime, t0), 4 × 6 h | yes |
| 55 | `sofa_renal_slope` | trend of renal SOFA (creatinine) over the first 24 h | sofa_trajectory | 6-hourly renal SOFA (creatinine only) | [intime, t0), 4 × 6 h | yes |
| 56 | `sofa_renal_onset_win` | first 6-hour window with renal dysfunction (creatinine) | sofa_onset | 6-hourly renal SOFA (creatinine only) | [intime, t0), 4 × 6 h | yes |
| 57 | `sofa_neuro_delta` | change in neurological SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly neurological SOFA | [intime, t0), 4 × 6 h | yes |
| 58 | `sofa_neuro_slope` | trend of neurological SOFA over the first 24 h | sofa_trajectory | 6-hourly neurological SOFA | [intime, t0), 4 × 6 h | yes |
| 59 | `sofa_neuro_onset_win` | first 6-hour window with neurological dysfunction | sofa_onset | 6-hourly neurological SOFA | [intime, t0), 4 × 6 h | yes |
| 60 | `sofa_total_first` | total SOFA in hours 0-6 | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 61 | `sofa_total_last` | total SOFA in hours 18-24 | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 62 | `sofa_total_max` | highest 6-hourly total SOFA | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 63 | `sofa_total_delta` | change in total SOFA, hours 0-6 to 18-24 | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 64 | `sofa_total_slope` | trend of total SOFA over the first 24 h | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 65 | `sofa_total_auc` | cumulative total SOFA burden over the first 24 h | sofa_trajectory | 6-hourly total SOFA | [intime, t0), 4 × 6 h | yes |
| 66 | `sofa_n_failed_organs` | number of organs with dysfunction in the first 24 h | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 67 | `sofa_time_to_first_failure_win` | first 6-hour window with any organ dysfunction | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 68 | `sofa_first_fail_resp` | respiratory dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 69 | `sofa_first_fail_cardio` | cardiovascular dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 70 | `sofa_first_fail_hepatic` | liver dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 71 | `sofa_first_fail_coag` | coagulation dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 72 | `sofa_first_fail_renal` | renal dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 73 | `sofa_first_fail_neuro` | neurological dysfunction among the first to appear | sofa_onset | 6-hourly SOFA sub-scores | [intime, t0), 4 × 6 h | yes |
| 74 | `sofa_resp_x_cardio` | respiratory x cardiovascular SOFA interaction | sofa_interactions | sofa_resp x sofa_cardio | [intime, t0) | yes |
| 75 | `sofa_cardio_x_renal` | cardiovascular x renal SOFA interaction | sofa_interactions | sofa_cardio x sofa_renal | [intime, t0) | yes |
| 76 | `sofa_renal_x_hepatic` | renal x liver SOFA interaction | sofa_interactions | sofa_renal x sofa_hepatic | [intime, t0) | yes |
| 77 | `sofa_neuro_x_resp` | neurological x respiratory SOFA interaction | sofa_interactions | sofa_neuro x sofa_resp | [intime, t0) | yes |
| 78 | `sofa_coag_x_hepatic` | coagulation x liver SOFA interaction | sofa_interactions | sofa_coag x sofa_hepatic | [intime, t0) | yes |
| 79 | `sofa_resp_x_renal` | respiratory x renal SOFA interaction | sofa_interactions | sofa_resp x sofa_renal | [intime, t0) | yes |

## 7. SOFA as implemented in the internship (frozen as-is, D-005)

A hand-rolled Python port of `sql/06_extract_sofa.sql` / `sql/07_extract_sofa_windows.sql`.
Those SQL files are not in the snapshot. CLAUDE.md said "SOFA from `mimiciv_derived.sofa`";
that is not what the code does. Deviations from standard SOFA and from mimic-code's
`mimiciv_derived.sofa`:

1. **Dobutamine is ignored.** Its rows are collected (`VASO_DOBU`) but never scored; any
   dobutamine should give cardiovascular 2.
2. **Respiratory:** the worst PaO2 and the highest FiO2 are paired even when measured at
   different times. Scores 3–4 do not require mechanical ventilation. When PaO2 is
   missing, a non-standard SpO2/FiO2 conversion `((SpO2/FiO2) − 64)/0.84` is used.
3. **Cardiovascular:** uses mean MAP rather than minimum. Vasopressin scores 2. Rates are
   taken as charted, without unit normalisation. Several itemids are CareVue ones that
   do not exist in MIMIC-IV (harmless).
4. **Missing values score 0** (no imputation). `urine_24h` and `bilirubin_max` are 0 when
   nothing was charted, so "not measured" and "zero" look the same.
5. **Windowed renal score** uses creatinine only. Gaps between 6-hour windows are carried
   forward; leading gaps are 0.
6. **Urine output adds irrigation fluid** (found in Stage 2). Item 227488 (GU irrigant volume
   in) is summed as urine, but mimic-code subtracts it. `urine_24h` and the renal score
   therefore overstate urine for irrigated patients; the demo has 32 such rows. Stage 2
   subtracts it (D-020).
7. **GCS for intubated patients** (found in Stage 2). The verbal score "No Response-ETT" is
   stored with value 1 and summed as such, so intubated patients get low GCS totals. In the
   demo, 41 % of verbal assessments are ETT. This affects the `gcs_verbal` mean feature and
   the neurological SOFA score; mimic-code scores these patients as 15 instead. You kept the
   internship convention, in Stage 2 as well (D-018).

## 8. Splits

| | FaTE-ICU (every model arm) | Internship (B0-faithful only) |
|---|---|---|
| Method | seeded hash of `subject_id` | numpy permutation of the stay rows |
| Patient-level | **yes** | no: stays of one patient can straddle train and test |
| Partitions | train 70 % / val 10 % / test 20 % | 10 repeats of train 80 % / test 20 %, no validation |
| Seed | 666 | 666 + repeat |
| Fit on train only | encoding, scaling, threshold (D-003, D-007) | encoding, scaling and Q3 fit on the whole cohort |

Rule: `u = first 8 bytes of sha256("666:<subject_id>") / 2^64`. Train if u < 0.7,
val if u < 0.8, else test.

## 9. B0 (internship model)

- Four model families in the internship: logistic/linear regression, RBF SVM,
  random forest (1000 trees) and XGBoost (3000 rounds). All use library defaults except
  those counts; there is no tuning.
- The two-stage framework uses an RF or XGBoost prior. The probability threshold that
  maximises √(TPR·TNR) is chosen on train scores, then LOS is regressed on the stays
  predicted short.
- FaTE-ICU's B0 is the XGBoost two-stage model on `fate_static` with the FaTE split.
  B0-faithful is the legacy scripts run as-is in the lab.
- Metrics:
  - classification: accuracy, F1, balanced accuracy, MCC, AUROC (positive = long), plus
    AUPRC and Brier in FaTE-ICU;
  - regression: RMSE, MAE, MAPE and R², where **R² = squared Pearson correlation**.

## 10. Reproduction targets (lab, MIMIC-IV v3.1)

Source: `legacy_los/README.md`. The original metrics are lost; these are the only
surviving internship numbers.

| Step | Stays |
|---|---|
| ICU stays | 94,458 |
| after 48 h readmission exclusion | 89,952 |
| after death-in-ICU exclusion | 83,041 |
| after LOS 1–21 d | 64,742 |
| after age ≥ 18 | 64,742 |
| final (complete case + vital ranges) | 55,394 |

Whole-cohort Q3 = 4.11 d; long-stay share = 24.9 %. Tolerance: exact counts, Q3 ± 0.01 d,
share ± 0.002 (`tests/test_realdata_cohort.py`).

## 11. Where the CLAUDE.md description differed from the code

| CLAUDE.md said | The code does |
|---|---|
| "SQL feature extraction" | pandas over raw CSVs; the SQL file is documentation |
| "SOFA features from `mimiciv_derived.sofa`" | hand-rolled Python SOFA (§7) |
| "SOFA-trajectory and care-response feature families" | trajectory, onset order and interaction families; no care-response features |
| "two-stage XGBoost pipeline" | four model families; two-stage with an RF or XGBoost prior |
| "41 unit tests" | 50 tests (the README count predates the ablation, aggregate and parallel tests) |
