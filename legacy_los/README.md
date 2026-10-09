# Faithful replication of Hempel et al. (2023)

**"Prediction of Intensive Care Unit Length of Stay in the MIMIC-IV Dataset"**
(Hempel, Sadeghi, Kirsten. *Appl. Sci.* 2023, 13, 6930. DOI 10.3390/app13126930)

This project reproduces the paper's two-stage (classification → regression) ICU
length-of-stay pipeline as faithfully as possible, then extends it with SOFA
severity features for two ablation studies.

* **Source of truth:** the paper PDF (`applsci-13-06930.pdf`).
* **Code reference:** the authors' R code,
  <https://gitlab.com/ul-mds/data-science/length-of-stay-prediction-mimic-iv>.
* Where the paper and the reference code disagree, the resolution is documented
  below and in code comments.

The sibling `icu_los_prediction/` project is **not** a faithful replication (it
uses a 3-class target, 30-day cap, hyperparameter grids, extra models, standard
scaling, median imputation, °C conversion, and extra features). It is reused here
only for (a) the raw MIMIC-IV CSVs, (b) the CSV-chunking idea, and (c) the
pure-Python SOFA scorer — each re-verified against the paper + R code.

---

## What the three run scripts do

| Script | Feature set |
|---|---|
| `scripts/run_hempel.py` | **Faithful baseline** — the paper's features only. Reproduces Tables 3–6. |
| `scripts/run_hempel_sofa.py` | Baseline **+ static SOFA** (`sofa_total`, 6 component sub-scores, `urine_24h`, `bilirubin_max`). |
| `scripts/run_hempel_sofa_dynamics.py` | Baseline **+ exactly 3 SOFA-dynamics groups**: (1) trajectory/velocity (Δ + slope over first-24h 6h windows), (2) organ-failure onset sequence, (3) sub-score interaction terms. |

`scripts/run_ablation.py` runs the two paired studies: **Hempel vs Hempel+SOFA**
and **Hempel vs Hempel+SOFA-dynamics** (identical splits; per-model metric deltas
+ paired significance). It **reuses the per-repeat metrics the three run scripts
already saved** (no model refitting), so it must be run *after* them and finishes
in seconds. A signature check guards that the compared runs scored the identical
set of stays.

---

## Faithful design decisions (paper + R code)

**Cohort / inclusion–exclusion** (from `Extraction.sql` + `data_preparation.R`):
* Exclude ICU readmissions within **48 h** of a prior ICU discharge in the same
  hospital admission.
* Exclude patients who **died during the ICU stay** (`deathtime ≤ outtime`).
* Keep ICU stays with **LOS > 1 day and ≤ 21 days**.
* **Adults** (`anchor_age ≥ 18`) — the paper says ">18"; the R code applies no age
  filter; MIMIC-IV is adults-only, so this is a confirmed no-op that reconciles
  both (verified: age filter removed 0 rows).
* **Complete-case**: drop any stay missing a feature (R `complete.cases`).
* Vital **mean** cleaning to Table-2 ranges (applied to the per-stay average, as in
  the R code): HR [25, 225], respiration (7, 40), saturation (50, 120),
  temperature [86, 113] °F.

**Features** (paper Table 1): age, gender; first care unit, admission type,
admission location, top-2 diagnoses; 7 vital-sign means (heart rate, O₂ sat,
respiration, temperature, GCS eye/verbal/motor); lab means. All temporal features
are the **mean over the first 24 h**.

**Models** (paper §2.5) — the four models only, **default hyperparameters, no
tuning**: logistic/linear regression, SVM (RBF), random forest (`n_estimators=1000`),
XGBoost (`n_estimators=3000`). RF trees and XGB rounds are the only non-defaults,
taken from the R code.

**Targets**: regression = `round(los, 2)`; classification = short (≤ Q3) vs long
(> Q3), dichotomised at the **3rd quartile of LOS** (paper's method).

**Evaluation**: **10 repeated random 80/20 splits**, report the mean. Classification
= accuracy, F1, balanced accuracy, MCC, ROC-AUC (positive class = long). Regression
= RMSE, MAE, MAPE, and **R² = squared Pearson correlation** (as in the R code).

**Two-stage framework** (paper §3.3): prior RF/XGB classifier, threshold chosen to
maximise the **G-mean** `√(TPR·TNR)`; then regress LOS on the **predicted-short**
subset (Tables 5–6).

### Resolved paper-vs-code conflicts

| Topic | Paper | R code | Our choice (documented) |
|---|---|---|---|
| Lab features | 19 (incl. Phosphate) | **18** (no Phosphate 50970) | **18** — matches the code that produced the results *(user-confirmed)* |
| Classification "LR" | "logistic regression" | `lm()` (linear prob. model) | **sklearn `LogisticRegression(penalty=None)`** — matches the paper's name; `penalty=None` = unregularized (glm-style) logistic regression, since sklearn's default L2/C=1 has no R counterpart. The repo has no classification-eval code, so `lm()` never produced Table 3 there *(recommended & confirmed)* |
| Regression R² | R² | `cor(actual,pred)²` | **corr²** (faithful to the code) |
| Class threshold | 3rd quartile (~4 d) | hardcoded 4 | **computed Q3** (came out **4.11 d** on our data) |
| LOS rounding | 1 dp | `round(los, 2)` | **2 dp** (code) |
| Temperature | — | °F, range-cleaned in °F | **°F, no conversion** |
| SVM probability | — | `probabiliy=TRUE` typo → never enabled; metrics from hard `type="response"` | **`probability=False`** — hard metrics from `predict()`, AUC from `decision_function` (rank-identical to Platt probabilities); matches the R code and is ~5–7× faster |
| SVM `gamma` | — | e1071 default `gamma = 1/p` | **`gamma='auto'`** (= 1/n_features) — matches e1071; sklearn's default `'scale'` (= 1/(p·Var(X))) would differ |
| SVM internal scaling | — | e1071 default `scale=TRUE` (z-scores features, **and the target `y` for regression**) | **`StandardScaler` on features** (SVC & SVR) **+ on the target** (SVR via `TransformedTargetRegressor`) — sklearn does no internal scaling, which otherwise makes SVR's `epsilon=0.1` mean 0.1 days instead of 0.1 SD |
| RF **regressor** defaults | "default values" | R `randomForest` regression defaults `mtry = p/3`, `nodesize = 5` | **`max_features=1/3`, `min_samples_leaf=5`** — match R's *regression* defaults (sklearn's defaults are all-features / leaf-1, which both deviate and are ~8× slower). RF *classifier* already matches R (`√p`, leaf 1). |
| MIMIC-IV version | v2.1 | v2.1 | **v3.1** (only version available locally) — see note below |

### R → Python translation notes (audited)

- **Model defaults:** every model was audited parameter-by-parameter against its
  *R library* default (not sklearn's). XGBoost matches (same library); `lm`/
  `LinearRegression` match (OLS); the SVM and RF-regressor gaps above are fixed.
- **Label encoding:** R `factor()→unclass()` gives 1-based integer codes in sorted
  level order; `pd.Categorical(...).codes` gives 0-based codes in the same sorted
  order. After the global min-max scaling both map to identical [0,1] values.
- **Rounding:** R `round()` and NumPy/pandas both use round-half-to-even, so
  `round(los, 2)` matches.
- **Repeated splits:** R's `sample()` RNG stream differs from NumPy's, so the exact
  train/test partitions are not identical, but the procedure (10 random 80/20
  splits, report the mean) is faithful; the paper itself averages to remove this.
- **min-max constant columns:** R would yield NaN (divide-by-zero); we map a
  constant column to 0 (robustness only — no such column arises in practice).
- **ICD-9→10:** R `touch::icd_map` wraps the CMS GEM; we use the CMS 2018 forward
  GEM directly with the same prefix-reduction + 22-chapter logic.

---

## Data & guards

Raw MIMIC-IV v3.1 CSV(.gz) are read from `data.raw_dir` in `config.yaml`
(default: `../icu_los_prediction/los_prediction/data/raw`, override with the
`MIMIC_RAW_DIR` env var). The large event tables (`chartevents`, `labevents`, and
for SOFA `inputevents`/`outputevents`) are streamed in chunks.

Every script **guards on required raw tables up front and stops with a clear
message if any are missing**; the SOFA variants additionally require
`inputevents`/`outputevents` and fail if the resulting SOFA components carry no
signal (a degenerate-feature guard).

**ICD-9 → ICD-10 conversion** uses the CMS 2018 forward GEM crosswalk
(`data/icd9to10gem/2018_I9gem.txt`) — the same mapping the R package `touch`
wraps. If it is absent, `diagnoses.py` stops with download instructions. Obtain it
from the CDC:
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2018/2018_I9gem.txt`
(sanity check: ICD-9 `5715 → K740/K7460/K7469`, the paper's §2.3 example, which
reduces to chapter `K00-K93`).

---

## How to run

```bash
pip install -r requirements.txt

# 1. Build the per-stay feature tables from the raw CSVs (writes data/processed/*.parquet)
python scripts/build_features.py --which all          # or: hempel | sofa | dynamics
#    quick wiring smoke on a small cohort:  --limit 2000

# 2. Run each pipeline (writes results/<name>/tables/*.csv + *.tex, incl. per-repeat metrics)
python scripts/run_hempel.py
python scripts/run_hempel_sofa.py
python scripts/run_hempel_sofa_dynamics.py

# 3. Ablation studies (reuses step 2's saved per-repeat metrics; no refitting; seconds)
python scripts/run_ablation.py
```

`--fast` on any run script shrinks the model sizes/repeats for a quick smoke test
**only** — never use it for reported numbers. The faithful defaults (RF 1000 trees,
XGB 3000 rounds, SVM with probability, 10 repeats on the full cohort) are
compute-heavy by design; a full run takes hours.

---

## Verification

* **Unit tests** (`tests/`, 41 tests, all passing): 22-chapter ICD mapping + GEM
  prefix-reduction; metrics (R² = corr², MAPE, balanced accuracy); SOFA threshold
  scoring + dynamics + interactions; cohort readmission logic; preprocessing
  (label-encode, min-max, Q3 target, complete-case, range cleaning); stepwise
  G-mean + confusion. Run with `python -m pytest tests/`.
* **Cohort build (real MIMIC-IV v3.1):** 94,458 ICU stays → 89,952 (−48h readmit)
  → 83,041 (−death-in-ICU) → 64,742 (LOS 1–21) → **55,394** after complete-case +
  range cleaning.
* **Key faithfulness signals match the paper closely:** dichotomisation
  threshold = **Q3 = 4.11 days** (paper ≈ 4); **long-stay prevalence = 24.9 %**
  (paper 75.2 % short / 24.8 % long, the "3:1 imbalance").
* **Metrics land on the paper** (real data): e.g. regression SVM MAE/MAPE
  1.67 / 45.6 (paper 1.68 / 48.4), RF RMSE 2.84 (paper 2.81), LR RMSE 2.86
  (paper 2.89); classification RF best accuracy ~0.79–0.81.

### Why our cohort (55,394) is larger than the paper's (41,473)

The paper used **MIMIC-IV v2.1**; only **v3.1** is available here, which contains
substantially more ICU stays (94,458 vs the paper's starting set). After the
identical filters the proportions match the paper (Q3 ≈ 4 d, ~25 % long), but the
absolute count is larger. This is the single unavoidable deviation and is purely a
dataset-version effect, not a pipeline difference.

---

## Project layout

```
config.yaml                 # all faithful settings (no tuning)
sql/Extraction.sql          # annotated copy of the GitLab SQL (optional Postgres path)
data/icd9to10gem/           # CMS 2018 I9gem crosswalk (external asset; guarded)
src/
  config.py                 # config + path resolution + data guards
  data/loader.py            # chunked CSV(.gz) reader
  features/                 # cohort, vitals, labs, demographics, administrative,
                            #   diagnoses (ICD9->10 + 22 chapters), SOFA static/windows/dynamics
  preprocessing/prepare.py  # dup-drop, complete-case, range clean, label-encode, min-max, targets
  models/                   # classifiers, regressors (defaults), stepwise G-mean
  evaluation/               # metrics (R²=corr²), repeated-split experiment runner
  ablation/compare.py       # paired deltas + significance
  pipeline.py               # end-to-end runner + reporting
scripts/                    # build_features, run_hempel[_sofa[_dynamics]], run_ablation
tests/                      # unit tests
```
