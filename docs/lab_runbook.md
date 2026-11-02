# Lab runbook (college lab desktop, RTX A2000 12 GB)

Each stage adds a section: prerequisites, exact commands, expected runtime on the lab
machine, and what to paste back. Commands use `uv run` and `python -m` entry points, so
they work in PowerShell, cmd and bash alike. Paste back **aggregate output only**. Every
script here already prints only counts (≥ 10), shapes, metrics and summary statistics.
Never copy rows, identifiers, free text, images or files off the lab machine.

Runtimes are rough estimates for a CPU-bound desktop; the GPU is unused until Stage 3.

---

## Stage 0: database, legacy reproduction, contract checks

### Prerequisites

- PhysioNet credentialed access to MIMIC-IV v3.1 and the files downloaded, e.g.
  `wget -r -N -c -np --user <USERNAME> --ask-password https://physionet.org/files/mimiciv/3.1/`.
- PostgreSQL ≥ 14 with `psql`, git, uv; about 150 GB free disk.
- mimic-code **v3.0.1**: `git clone --branch v3.0.1 --depth 1 https://github.com/MIT-LCP/mimic-code.git`
- The public CMS GEM file `2018_I9gem.txt` (see `docs/laptop_setup.md` §2).
- The repository cloned and set up:

  ```
  git clone <repo-url> fate-icu
  cd fate-icu
  uv sync
  ```

  Then copy `.env.example` to `.env` and set:
  - `FATE_MACHINE=lab`
  - `FATE_DB_URL=postgresql+psycopg://<user>:<password>@localhost:5432/mimiciv`
  - `MIMIC_RAW_DIR=<path>/mimiciv/3.1` (the folder holding `hosp/` and `icu/`)
  - `ICD9_GEM_PATH=<path>/2018_I9gem.txt`

### 0.1 Load MIMIC-IV v3.1 and build the derived concepts (≈ 3–6 h)

Same scripts and schema names (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`) as the
laptop. On Linux, mimic-code's helper does all steps, including concepts and validation:

```
cd mimic-code
createdb mimiciv
PGDATABASE=mimiciv mimic-iv/buildmimic/postgres/build_mimic.sh <path>/mimiciv/3.1
```

By hand (any OS; on Windows put Git's `usr\bin` on PATH for `gzip`):

```
createdb mimiciv
psql -d mimiciv -f mimic-iv/buildmimic/postgres/create.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=<path>/mimiciv/3.1 -f mimic-iv/buildmimic/postgres/load_gz.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=<path>/mimiciv/3.1 -f mimic-iv/buildmimic/postgres/constraint.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=<path>/mimiciv/3.1 -f mimic-iv/buildmimic/postgres/index.sql
cd mimic-iv/concepts_postgres
psql -d mimiciv -v ON_ERROR_STOP=1 -f postgres-functions.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -f postgres-make-concepts.sql
```

Check: `psql -d mimiciv -c "select count(*) from mimiciv_icu.icustays" -c "select count(*) from mimiciv_derived.sofa"`
(94,458 ICU stays expected on v3.1).

### 0.2 All tests, including `realdata` (≈ 1–2 h the first time; the tables are cached)

```
uv run pytest
```

`realdata` tests build the contract cohort from PostgreSQL and compare it with the
internship's recorded flow (94,458 → 89,952 → 83,041 → 64,742 → 55,394 stays,
whole-cohort Q3 4.11 d, 24.9 % long). `demo` tests are skipped in the lab.

### 0.3 Reproduce the internship pipeline as-is (B0-faithful; many hours)

The internship's metrics were lost, so this run produces the reference numbers. Copy the
GEM file to `legacy_los/data/icd9to10gem/2018_I9gem.txt` (git-ignored); the legacy code
looks for it there. Don't change `legacy_los/config.yaml` between these scripts.

```
cd legacy_los
uv run python scripts/build_features.py --which all
uv run python scripts/run_hempel.py
uv run python scripts/run_hempel_sofa.py
uv run python scripts/run_hempel_sofa_dynamics.py
uv run python scripts/run_ablation.py
cd ..
```

Rough runtimes: `build_features` ≈ 1–2 h (streams the CSVs). Each `run_*` script takes
several hours: 10 repeats × 4 models × 4 experiments, RF 1000 trees, XGB 3000 rounds.
`run_ablation` takes seconds. If the machine may sleep or reboot, run
`uv run python scripts/run_all_resumable.py` instead of the three `run_*` scripts. It
runs them in order with per-model checkpoints and skips finished work on restart.
Outputs go to `legacy_los/results/` and `legacy_los/data/processed/`, both git-ignored.

### 0.4 PostgreSQL vs legacy CSV reader on the full data (≈ 2–3 h)

```
uv run python -m fate_icu.tabular.crosscheck --config configs/tabular/full.yaml
```

### 0.5 Freeze the long-stay threshold (minutes, uses the cache)

```
uv run python -m fate_icu.tabular.freeze_threshold --config configs/tabular/full.yaml
```

Paste the output back. Claude writes `train_q3_days` into `configs/contract.yaml` in a
reviewed commit; pull that commit before 0.6.

### 0.6 B0 on the FaTE split and the storetime audit (≈ 30–60 min)

```
uv run python -m fate_icu.tabular.run --config configs/tabular/full.yaml
uv run python -m fate_icu.tabular.audit_leakage --config configs/tabular/full.yaml
```

### Paste back

1. The PostgreSQL row counts from 0.1.
2. The final `pytest` summary line, and any failure messages.
3. The console summaries of the three `run_*` scripts, `run_ablation.py`, and the
   `run_meta` table of each run (`legacy_los/results/<name>/tables/run_meta.csv`).
4. The JSON printed by `crosscheck`, `freeze_threshold`, `run` and `audit_leakage`.

---

## Stage 2: hourly time-series tensors

### Prerequisites

- Stage 0 step 0.1 is done. The database must have the derived concepts, including
  `mimiciv_derived.sofa` and `mimiciv_derived.norepinephrine_equivalent_dose`.
- The contract cohort is cached under `artifacts/tabular/` by Stage 0 steps 0.2 or 0.6. If it
  is not, the first build creates the cache, which adds about 1–2 h.

### Build (about 30–60 min)

```
uv run python -m fate_icu.data.timeseries.build --config configs/timeseries/full.yaml
```

Runtime breakdown:
- SQL extraction: 11 batches of 5,000 stays against chartevents, labevents, outputevents and
  the two derived tables. This is most of the time.
- Grid and tensors: a few minutes.

Output: `artifacts/timeseries/<run_id>/`, about 55,000 × 24 × 30 per array, a few hundred MB
compressed. Note the `run_id` for Stage 3.

The build stops, rather than writing tensors, in two cases:
- an input dated after t0 is found;
- a channel has no train value or is constant.

If that happens, paste the message.

### Paste back

The printed JSON:
- the run id and shape;
- rows by source (seen, dropped by each rule, accepted). For `sofa`, "accepted" counts
  component scores, not rows;
- implausible values dropped per channel;
- missingness per channel;
- `normaliser_warnings` (expected: empty);
- `tensor_sha256`;
- the comparison with the internship features.

All of it is aggregates, with counts below 10 suppressed.

---

## Stage 1: link MIMIC-IV and MIMIC-CXR

### Prerequisites

- Your own credentialed access to **MIMIC-CXR-JPG 2.1.0** on PhysioNet. This needs CITI
  "Data or Specimens Only Research" training and the PhysioNet Credentialed Health Data
  Use Agreement 1.5.0, signed for this dataset.
- Stage 0 done in the lab: the database and the cached contract cohort.
- Disk space. My rough estimate is about 1.5 MB per selected image, e.g. about 20 GB for
  15,000 images.
- `wget`. On Windows use Git Bash, WSL, or `winget install JernejSimoncic.Wget`.
- In `.env`, set `MIMIC_CXR_DIR=<parent>/mimic-cxr-jpg/2.1.0`. This folder will hold the
  metadata file and `files/`.

### 1.1 Download the metadata (minutes)

Run this from `<parent>`. `--cut-dirs=1` recreates `mimic-cxr-jpg/2.1.0/...` there.

```
wget -r -N -c -np -nH --cut-dirs=1 --user <PHYSIONET_USERNAME> --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
```

### 1.2 Link (minutes)

```
uv run python -m fate_icu.data.cxr.link --config configs/cxr/full.yaml
```

Output goes to `artifacts/cxr/<run_id>/`: `paired_cohort.parquet`, `image_list.txt`
(the selected images only) and `report.json`. The command stops if a required metadata
column is missing, or if a selected image is dated after t0.

**Check before going on:**
- If `below_min_paired_train_stays` is `true` (fewer than 2,000 paired train stays), stop
  and tell me. Stage 3 should not run; the fallback is report text (Stage 8).
- If `date_alignment.share_with_an_xray_during_an_icu_stay` is near zero, the X-ray and ICU
  timelines don't line up. Stop and tell me.

### 1.3 Download only the selected images (hours)

Run from `<parent>` again, giving the full path to the list:

```
wget -r -N -c -np -nH --cut-dirs=1 --user <PHYSIONET_USERNAME> --ask-password -i <repo>/artifacts/cxr/<run_id>/image_list.txt --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

The images land in `MIMIC_CXR_DIR/files/...`. If interrupted, run the same command again:
`-N -c` skips or resumes what is already there. Never download the full image set.

### 1.4 Build the image cache (about 1 h for 15,000 images, CPU)

```
uv run python -m fate_icu.data.cxr.cache --config configs/cxr/full.yaml --link-run <run_id>
```

The cache resumes where it stopped. It exits with an error if an image is missing or
unreadable; their keys are listed only in `artifacts/cxr_cache_runs/<run>/problems.txt`. If
that happens, repeat step 1.3, then this step.

### Paste back

1. The JSON printed by `link`, covering:
   - the flow counts;
   - the shares linked by split and by anchor year group;
   - the paired train count and its flag;
   - `hours_before_t0`;
   - LOS for paired versus unpaired stays;
   - the views;
   - the date alignment;
   - `images_to_download`.
2. The JSON printed by `cache`.
