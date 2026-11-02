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
