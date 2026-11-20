# Lab runbook (college lab desktop, RTX A2000 12 GB)

> **2026-11-20:** the college lab is not available. "Lab" in this file now means the GCP VM described in the
> last section, "GCP: the run machine" (D-049). Use `FATE_MACHINE=gcp`. Runtimes written for the A2000 are guides.

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

---

## Stage 5: RAG index (public data; laptop, then a Kaggle or Colab GPU, then the lab)

Stage 5 uses only public text, so it is built on the laptop now. A Kaggle or Colab GPU runs
the model-heavy step. The lab only receives the finished index and checks it.

### Prerequisites (laptop)

- In `.env`:
  - `FATE_RAG_DIR=D:\fate-data\rag`, or any folder outside OneDrive with about 10 GB free;
  - `NCBI_EMAIL=<your email>`: NCBI's usage policy asks for it, and it is sent only to NCBI;
  - optionally `NCBI_API_KEY=<key>`, free from your NCBI account (Account settings > API Key
    Management); it makes the download about 3 times faster.
- Download the four guideline PDFs into `<FATE_RAG_DIR>\raw\guidelines\` under these names:

  | File | Guideline | Where (DOIs checked against Crossref on 2026-11-13) |
  |---|---|---|
  | `ssc2021.pdf` | Surviving Sepsis Campaign 2021 (Evans et al., Intensive Care Med 2021) | https://doi.org/10.1007/s00134-021-06506-y |
  | `kdigo_aki_2012.pdf` | KDIGO Clinical Practice Guideline for Acute Kidney Injury (2012) | https://kdigo.org/guidelines/acute-kidney-injury/ |
  | `esicm_ards_2023.pdf` | ESICM guidelines on ARDS (Grasselli et al., Intensive Care Med 2023) | https://doi.org/10.1007/s00134-023-07050-7 (Crossref lists CC BY-NC 4.0) |
  | `padis_2018.pdf` | PADIS guideline (Devlin et al., Crit Care Med 2018) | https://doi.org/10.1097/CCM.0000000000003299 |

  The command below creates the folder and shows which files are still missing and which
  licences are still unconfirmed; run it again after saving the files:

  ```
  uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source guidelines
  ```

  Then, in `configs/rag/full.yaml`, replace each "TO CONFIRM" licence with the licence shown
  on the publisher's page, and check each citation against its PDF.
- The MedCPT models (about 1.3 GB) are already in the laptop's Hugging Face cache.

### 5.1 Download StatPearls and PubMed (laptop; about 9 + 12 + 19 min measured; all resume)

```
uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source statpearls --check-size
uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source statpearls
uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source pubmed --ids-only
uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source pubmed
```

If the size check fails, NCBI has updated StatPearls. Review the change, then set
`sources.statpearls.expected_size` in the config.

The `--ids-only` run (about 5-20 min) runs every PubMed search and downloads no records. It
reports how many records each search found, the total, and `searches_with_pubmed_notes`: any
phrase or field PubMed did not recognise (an unknown MeSH heading never matches and raises
no error). Check these before the last command downloads the records. Finished searches are
cached, so the last command does not repeat them. NCBI asks that bulk requests run outside
US Eastern working hours (9 PM to 5 AM on weekdays, or at weekends); the commands print a
time-stamped line per search and per 25 batches, so a background run can be followed.

### 5.2 Corpus, BM25 and the GPU bundle (laptop, about 20 min)

```
uv run python -I -m fate_icu.rag.build_corpus --config configs/rag/full.yaml
uv run python -m fate_icu.rag.build_index --config configs/rag/full.yaml --part bm25
uv run python -m fate_icu.rag.gpu_bundle --config configs/rag/full.yaml
```

`gpu_bundle` writes four files into `<FATE_RAG_DIR>\rag_full\gpu_bundle\` and prints the
three notebook commands:
- the code as a wheel;
- `passages.parquet`;
- `gpu_spec.json`;
- `bm25_hits.json`.

### 5.3 The GPU step on Kaggle (about 1-2 h on a T4)

1. Kaggle: Create > New Dataset. Upload the four bundle files, keep the dataset private, and
   note its name.
2. Create > New Notebook. In the right-hand panel:
   - Accelerator: GPU T4 or P100;
   - Internet: on (this needs a phone-verified account);
   - Add Input: your dataset.
3. Run the three printed commands, each in its own cell with a leading `!`. Replace
   `<your-dataset-name>` with your dataset's name.
4. Save Version > Save & Run All (Commit), so it runs without the browser open. When it has
   finished, open the version's Output tab and download the `gpu_out` folder.
5. Put it at `<FATE_RAG_DIR>\rag_full\index\<version>\gpu_out\`. The version is printed by
   `build_index` and `gpu_bundle`.

**Colab instead of Kaggle:**
1. Upload the four files to Google Drive.
2. Choose Runtime > Change runtime type > T4 GPU, and mount Drive.
3. Run the same commands with your Drive paths, and `--out /content/drive/MyDrive/fate-icu/gpu_out`.

### 5.4 Install and check the GPU output (laptop, about 5 min)

```
uv run python -m fate_icu.rag.build_index --config configs/rag/full.yaml --part dense
uv run python -m fate_icu.rag.verify --config configs/rag/full.yaml
```

The install refuses GPU output that does not match this laptop. It checks:
- the corpus hash and model of the embeddings;
- the BM25 lists, which must be identical;
- the dense lists, recomputed for every query;
- two queries' reranks, recomputed.

### 5.5 Relevance grades (you, about 3-6 h over several sittings)

```
uv run python -m fate_icu.rag.evaluate sheet --config configs/rag/full.yaml
uv run python -m fate_icu.rag.label --config configs/rag/full.yaml
uv run python -m fate_icu.rag.evaluate import --config configs/rag/full.yaml --rater <your initials>
uv run python -m fate_icu.rag.evaluate score --config configs/rag/full.yaml
```

How to grade is in `docs/rag_grading_guide.md`.

### 5.6 Copy the index to the lab (minutes)

Copy these from the laptop's `<FATE_RAG_DIR>\rag_full\` to the same place under the lab's
`FATE_RAG_DIR`, by USB drive or cloud storage (it is public data):
- `corpus\<corpus id>\`;
- `index\<version>\` (the `gpu_out\` folder can stay behind);
- `corpus.CURRENT` and `index.CURRENT`.

Then set `FATE_RAG_DIR` in the lab's `.env`, and run:

```
uv run python -m fate_icu.rag.verify --config configs/rag/full.yaml
```

### Paste back

The JSON printed by `verify` in the lab.

## Stage 3: encoders, fusion model, baselines

### Prerequisites

- Stage 0 done: the full database, the legacy reproduction and **the frozen long-stay threshold**
  in `configs/contract.yaml` (the full config refuses to start without it).
- Stage 2 run finished (its run id is under `artifacts\timeseries\`).
- Stage 1 link finished (`artifacts\cxr\<run id>\`) and its images downloaded.
- `.env` on the lab machine has `FATE_DEVICE=auto` (or `cuda`) and `MIMIC_CXR_DIR`.
- `uv sync` with CUDA 13 support for torch (D-029); check with
  `uv run python -c "import torch; print(torch.cuda.is_available())"`.

### 3.1 Download the image weights (about 28 MB, once)

```
uv run python -m fate_icu.models.image
```

It prints `densenet121-res224-chex: 1024 pooled features + 11 pathology logits = 1035 values per image`.
The lab needs internet for this; otherwise copy
`%USERPROFILE%\.torchxrayvision\models_data\chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
from a machine that has it.

### 3.2 Set the inputs in `configs/models/full.yaml`

Set `timeseries_run` to the Stage 2 run id and `cxr_run` to the Stage 1 link run id: the model
run then reuses them. Leave them `null` to rebuild both inside the run.

### 3.3 Train every arm, 5 seeds (several hours on the A2000: time one seed first)

```
uv run python -m fate_icu.models.run --config configs/models/full.yaml
```

It preprocesses and caches the selected images, caches their features, trains B0, B1 to B6,
M and the Transformer ablation for 5 seeds, and writes `outputs\models\<run id>\`
(`metrics.json`, `run_manifest.json`, and stay-level `predictions_<arm>_seed<k>.npz` for
Stages 4 and 7). The console shows aggregates only. To time one seed, copy the config and set
`n_seeds: 1`.

### 3.4 Tune lambda (the Huber weight) on validation

Run the config for `lambda_reg` in 0.1, 0.3, 1 and 3 with `arms: [M]` and `n_seeds: 1`. Pick
the value with the best validation AUPRC; test results are never used for this. Record the
choice in `docs/decisions.md`.

### Paste back

The JSON printed at the end of the run: the table of arms and populations (AUROC, AUPRC,
Brier, ECE, MAE, RMSE with their intervals), the split sizes, `stays_with_image`, and the
`constant_static_features` count. Also the time of one seed. A null gain from the image is an
acceptable result: say what the table shows.

## Stage 4: attribution and evidence packets

### Prerequisites

- The Stage 3 run finished with its checkpoints: `checkpoints\models\<run id>\M_seed<seed>.pt`
  (written by `fate_icu.models.run`).
- The inputs that run used still exist (the Stage 2 tensors, the Stage 1 link and the image
  cache), because the packets are built from the same arrays.
- `uv sync`, and the two weight files, which download on first use and need internet once:
  the image model (`python -m fate_icu.models.image`, 28 MB) and the segmentation model (273 MB,
  downloaded when the first image driver is described).
- The same `.env` as Stage 3.

### 4.1 Set the model run

In `configs/attribution/full.yaml` set `checkpoint_run` to the Stage 3 run id and `seed` to the
seed of the checkpoint to explain. `n_stays: null` explains every stay of the split;
`n_tau_stays: null` fits tau on every validation stay.

### 4.2 Build the packets (hours: time one split first)

```
uv run python -m fate_icu.attribution.build_packets --config configs/attribution/full.yaml
```

It fits tau on the validation stays, then writes one `packet_<stay id>.json` per stay of the
split under `outputs\attribution\<run id>\packets\` (credentialed-derived: never copy them off
the lab machine). Run it once with `split: val` and once with `split: test`. The slow parts
are the ExtremalMask training (one mask is learned for a batch of stays: time it) and Integrated Gradients and, for stays whose drivers
include an image unit, Grad-CAM and segmentation (several seconds each). To time it, set
`n_stays: 50`.

### 4.3 Check the packets (a few minutes)

The command prints aggregates only: the number of packets, tau, drivers per stay, drivers by
modality and direction, and how many image drivers were reported as findings. Look for:
- every packet valid (`schema_valid` equals `packets`);
- most stays with at least one driver, and a mean near 2 to 3;
- not all drivers from one modality;
- image findings reported for a minority of image drivers.

### 4.4 The semi-synthetic benchmark and the validation report (not built yet)

The spec also asks for a semi-synthetic benchmark (patterns planted in real tensors) and a
validation report on the real model. Neither is built: say so when you start the lab runs,
and I will write them against the real checkpoint's shapes.

### Paste back

The JSON printed by `build_packets` for the `val` and `test` runs, and the time of the
50-stay trial.

## GCP: the run machine (replaces the college lab)

From 2026-11-20 every "lab" step in this file runs on a GCP VM in your credentialed project. Use the
same commands, with `FATE_MACHINE=gcp`. Runtimes marked "A2000" were estimated for the lab GPU; the
L4 (24 GB, bf16) is a little faster and has more memory. Time one seed first.

**Rules that do not change** (CLAUDE.md A3): the data and everything derived from it stay inside this
project. Nothing is copied to your laptop, Kaggle, Colab or git, no bucket or disk that holds it is
public, and you paste back aggregates only. PhysioNet's cloud rules can change: read its current page on
cloud use before loading the data, and keep to what it says.

### G1. One-time project setup (about 30 minutes)

You need a GCP project with billing on, and `gcloud` installed on the laptop (it only talks to GCP).
Replace `PROJECT`, `REGION` and `ZONE` (an L4 zone with spot capacity, for example `us-central1-a`).

```
gcloud config set project PROJECT
gcloud services enable compute.googleapis.com iap.googleapis.com
```

Set a billing **budget alert** in the console first (Billing > Budgets and alerts), so a forgotten VM cannot
run up a bill.

### G2. Network without a public address (about 10 minutes)

The VM gets no public IP; you reach it through Google's identity-aware proxy, and a NAT gives it outbound
internet (for PhysioNet, PyPI and GitHub).

```
gcloud compute firewall-rules create fate-allow-iap-ssh --network=default --direction=INGRESS --action=allow --rules=tcp:22 --source-ranges=35.235.240.0/20
gcloud compute routers create fate-router --network=default --region=REGION
gcloud compute routers nats create fate-nat --router=fate-router --region=REGION --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
```

### G3. Disk and VM (about 10 minutes)

A 500 GB data disk holds PostgreSQL (about 100 GB), the downloaded files, the image cache and the
checkpoints. It survives when the VM is stopped or reclaimed.

```
gcloud compute disks create fate-data --size=500GB --type=pd-balanced --zone=ZONE
gcloud compute instances create fate-icu --zone=ZONE --machine-type=g2-standard-8 --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced --disk=name=fate-data,device-name=fate-data,mode=rw,boot=no --no-address --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --metadata=enable-oslogin=TRUE
```

`g2-standard-8` comes with one L4 attached (no accelerator flag). If the zone has no spot L4 capacity the
command says so: try another zone. Roughly, the VM costs a few tenths of a dollar per hour while it runs
(spot) and the two disks about $60 a month whether it runs or not: check Google's pricing calculator.

Connect from the laptop:

```
gcloud compute ssh fate-icu --zone=ZONE --tunnel-through-iap
```

### G4. Prepare the VM (about 45 minutes, once)

On the VM. Check each step's output before the next.

```
# the GPU driver, with Google's documented installer
curl -L https://github.com/GoogleCloudPlatform/compute-gpu-installation/raw/main/linux/install_gpu_driver.py -o install_gpu_driver.py
sudo python3 install_gpu_driver.py
nvidia-smi
```

`nvidia-smi` must show an L4. Its "CUDA Version" line must be at least the one the locked torch needs
(D-029 says CUDA 13 for torch 2.15): if it is lower, tell me and I will pin an older torch for the VM.

```
# mount the data disk
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-fate-data
sudo mkdir -p /mnt/data && sudo mount -o discard,defaults /dev/disk/by-id/google-fate-data /mnt/data
echo '/dev/disk/by-id/google-fate-data /mnt/data ext4 discard,defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo chown $USER /mnt/data
```

(`mkfs` erases the disk: run it only this first time.)

```
# tools
sudo apt-get update && sudo apt-get install -y git tmux postgresql postgresql-contrib wget
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Move PostgreSQL's data onto the data disk, so the 100 GB database is not on the small boot disk:

```
sudo systemctl stop postgresql
sudo mkdir -p /mnt/data/postgresql && sudo rsync -a /var/lib/postgresql/ /mnt/data/postgresql/
sudo sed -i "s#^data_directory = .*#data_directory = '/mnt/data/postgresql/14/main'#" /etc/postgresql/14/main/postgresql.conf
sudo systemctl start postgresql
sudo -u postgres psql -c "show data_directory;"
```

Give your Linux user a PostgreSQL role, so `psql` works without a password over the local socket:

```
sudo -u postgres createuser --superuser $USER
```

Clone the repository (the repo is private: use a fine-grained, read-only access token that you type when
asked; never put it in a file in the repo) and install:

```
cd /mnt/data && git clone https://github.com/abhinaba01/multi-modal-icu-los-prediction.git fate-icu
cd fate-icu && uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Create `.env` from `.env.example` with `FATE_MACHINE=gcp`, `FATE_DB_URL=postgresql+psycopg:///mimiciv` (the local socket,
no password) , and `FATE_DATA_DIR`, `FATE_ARTIFACTS_DIR`, `FATE_OUTPUTS_DIR` and `FATE_CHECKPOINTS_DIR` under
`/mnt/data/fate-icu-data/...` so that nothing credentialed is written to the boot disk.

### G5. Load the data (several hours, mostly waiting)

One command does what Stage 0 step 0.1 describes. It runs only on the VM (it refuses to run on
Windows), and it takes your PhysioNet username; **wget asks you for the password itself**, so type it
there and nowhere else. Run it inside `tmux`:

```
tmux new -s fate                  # reattach later with: tmux attach -t fate
cd /mnt/data/fate-icu
uv run python -m fate_icu.utils.mimic_load --step all --user <PHYSIONET_USERNAME> --dest /mnt/data/mimic/mimiciv/3.1 --code-dir /mnt/data/mimic-code
```

The steps are `preflight` (tools, free disk, clones mimic-code at the pinned tag), `download`, `verify`
(every file against PhysioNet's checksum list), `load`, `concepts` and `check` (counts only; it fails if
the ICU stay count is not the 94,458 of MIMIC-IV v3.1). Add `--dry-run` first to see every command
without running it. To repeat one step, give it by name, for example `--step load`; a database that
already holds the stays is skipped unless you add `--force`, so re-running after an interruption is safe.
The download resumes where it stopped.

Then follow **Stage 0, steps 0.2 to 0.6**, and Stages 2 and 1, as written for the lab.

### G6. Running the stages, and what a reclaimed VM means

Order: Stage 0, 2, 1, 3, 4, then 6 and 7 when they exist. Start the VM only when you have a job; stop
it afterwards (a stopped VM costs only its disks):

```
gcloud compute instances start fate-icu --zone=ZONE
gcloud compute instances stop fate-icu --zone=ZONE
```

A spot VM can be reclaimed by Google at any time; it then stops and keeps its disks. To carry on:
start it, reconnect, `tmux new -s fate`, and repeat the command with `--resume`. Stage 3 saves each
arm as soon as it is trained, so it loses at most the arm in progress:

```
uv run python -m fate_icu.models.run --config configs/models/full.yaml --resume <run id>
```

The run id is printed at the start and is the folder name under `outputs/models/`. Resuming refuses to
continue if the config or the long-stay threshold changed.

### G7. Before you stop for good

- Stop the VM after each session; delete the VM and both disks when the project is finished.
- Do not create snapshots or images of the data disk unless they stay in the project.
- Keep the billing budget alert on.
- Copy nothing off the VM except the aggregate text that each stage asks you to paste back.
