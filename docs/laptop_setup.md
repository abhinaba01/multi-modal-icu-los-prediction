# Laptop setup (build phase, Windows)

Open data only. Run these yourself in **PowerShell**; Claude Code never downloads data
or installs software. Allow about 45 minutes; most of it is the demo database build.

Keep every data folder **outside the repository and outside OneDrive**. The examples
use `C:\mimic`.

## 1. Tools

1. **uv** (Python and package manager):

   ```powershell
   winget install --id=astral-sh.uv -e
   ```

   Open a new terminal afterwards so `uv` is on PATH. uv installs Python 3.11 for the
   project itself; the system Python 3.13 is not used.

2. **PostgreSQL** (16 or newer; this laptop runs 18): run the EDB installer from <https://www.postgresql.org/download/windows/>
   and keep "Command Line Tools" selected. Note the password you set for the `postgres` user.
   Add the tools to PATH for your user, then open a new terminal:

   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\PostgreSQL\18\bin", "User")   # your version's number
   ```

3. **gzip**: mimic-code's loader calls `gzip -dc`. Git for Windows already ships it in
   `C:\Program Files\Git\usr\bin`; step 3 adds that folder to PATH for the session.

4. *(Recommended)* Keep the virtual environment out of OneDrive, which would otherwise
   sync thousands of package files:

   ```powershell
   [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", "C:\venvs\fate-icu", "User")
   ```

## 2. Open data and scripts

1. **MIMIC-IV Clinical Database Demo v2.2** (open access, ODbL, 100 patients, 15 MB):
   on <https://physionet.org/content/mimic-iv-demo/2.2/> click "Download the ZIP file" and
   extract it so that `C:\mimic\mimic-iv-clinical-database-demo-2.2\hosp\` and `...\icu\`
   exist. Keep the `.csv.gz` files compressed.

2. **CMS 2018 ICD-9 → ICD-10 GEM** (public; the legacy diagnosis features need it):

   ```powershell
   New-Item -ItemType Directory -Force C:\mimic\icd9to10gem
   Invoke-WebRequest -Uri "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2018/2018_I9gem.txt" -OutFile C:\mimic\icd9to10gem\2018_I9gem.txt
   ```

3. **mimic-code v2.4.0**, the last release built for MIMIC-IV v2.2. The lab uses v3.0.1,
   built for v3.1.

   ```powershell
   git clone --branch v2.4.0 --depth 1 https://github.com/MIT-LCP/mimic-code.git C:\mimic\mimic-code-v2.4.0
   ```

## 3. Build `mimiciv_demo`

```powershell
$env:Path += ";C:\Program Files\Git\usr\bin"   # gzip for load_gz.sql
$env:PGUSER = "postgres"                       # psql asks for the password
$env:PGCLIENTENCODING = "UTF8"
$demo = "C:/mimic/mimic-iv-clinical-database-demo-2.2"

createdb mimiciv_demo
Set-Location C:\mimic\mimic-code-v2.4.0\mimic-iv\buildmimic\postgres
psql -d mimiciv_demo -f create.sql
psql -d mimiciv_demo -v ON_ERROR_STOP=1 -v mimic_data_dir=$demo -f load_gz.sql
psql -d mimiciv_demo -v ON_ERROR_STOP=1 -v mimic_data_dir=$demo -f constraint.sql
psql -d mimiciv_demo -v ON_ERROR_STOP=1 -v mimic_data_dir=$demo -f index.sql
```

Derived concepts (creates `mimiciv_derived`, including `sofa`). The scripts include
files by relative path, so run them from their own folder:

```powershell
Set-Location C:\mimic\mimic-code-v2.4.0\mimic-iv\concepts_postgres
psql -d mimiciv_demo -v ON_ERROR_STOP=1 -f postgres-functions.sql
psql -d mimiciv_demo -v ON_ERROR_STOP=1 -f postgres-make-concepts.sql
```

Check (the demo has 100 patients; expect roughly 140 ICU stays and a non-zero SOFA count):

```powershell
psql -d mimiciv_demo -c "select count(*) from mimiciv_icu.icustays" -c "select count(*) from mimiciv_derived.sofa"
```

If a step fails, paste the error message to Claude. It contains no patient data.

## 4. `.env`

```powershell
Set-Location "C:\Users\LENOVO\OneDrive\Desktop\MTech Project(Abhinaba)\multi_modal_icu_los_prediction"
Copy-Item .env.example .env
notepad .env
```

Fill in (URL-encode special characters in the password, e.g. `@` → `%40`):

```
FATE_MACHINE=laptop
FATE_DB_URL=postgresql+psycopg://postgres:<password>@localhost:5432/mimiciv_demo
MIMIC_DEMO_DIR=C:/mimic/mimic-iv-clinical-database-demo-2.2
ICD9_GEM_PATH=C:/mimic/icd9to10gem/2018_I9gem.txt
```

Leave `MIMIC_RAW_DIR` commented out on the laptop. Claude Code cannot read `.env`.

## 5. Project environment and checks

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run python -m fate_icu.tabular.run --config configs/tabular/smoke.yaml
uv run python -m fate_icu.tabular.audit_leakage --config configs/tabular/smoke.yaml
```

`pytest` runs the 50 legacy tests and the FaTE-ICU tests. The `demo` tests compare the
legacy CSV reader with PostgreSQL on the demo; they are skipped until `FATE_DB_URL` is set.

## 6. Moving to another computer

Written on 2026-11-15, before giving this laptop away. The whole project is in five places;
only the first is in git.

| # | What | Where it is on this laptop | Size | How it moves |
|---|---|---|---|---|
| 1 | Code, docs, configs, tests, `uv.lock` | the repository (about 200 files and the whole history) | 8 MB | `git push` to a **private** repository, then `git clone` |
| 2 | Stage 5 public downloads | `D:\fate-data\` (`rag\raw\pubmed`, `rag\raw\statpearls`, `rag\raw\guidelines`, `rag\rag_full`) | 4.6 GB | copy the folder, then verify it with its checksum list |
| 3 | The demo database and its sources | `C:\mimic\` and the PostgreSQL database `mimiciv_demo` | 29 MB + 215 MB | copy `C:\mimic\`; restore a dump or rebuild the database |
| 4 | The MedCPT models | `C:\Users\LENOVO\.cache\huggingface\hub\` | 1.3 GB | copy (optional: they download again in about 20 minutes) |
| 5 | Settings and secrets | `.env` in the repository | 1 KB | type it again on the new computer; never through git or a cloud folder |

Copy rather than re-download the Stage 5 data. PubMed grows every day, so the same searches
run tomorrow return other records, a different corpus hash and a different index. The 151,341
PMIDs of 2026-11-13 are pinned in `rag\raw\pubmed\pmids.json` and `esearch\`. NCBI also
replaces the StatPearls archive from time to time; the config pins its exact size and refuses
another file.

Not worth moving: the virtual environment (1.4 GB, rebuilt by `uv sync`), the uv cache
(1.5 GB), `artifacts\` (3.5 MB of caches that the smoke runs rebuild),
`D:\fate-data\rag\rag_dryrun_*` (240 MB of scratch corpus builds), `__pycache__`.
Nothing credentialed is on this laptop (checked 2026-11-15): the demo is open data.

### A. On this laptop

1. **Commit and push.** There is no git remote yet, so nothing has ever left this laptop. On
   github.com create a new repository named `fate-icu`, set it to **Private**, and leave
   "Add a README", ".gitignore" and "license" unticked. Then:

   ```powershell
   Set-Location "C:\Users\LENOVO\OneDrive\Desktop\MTech Project(Abhinaba)\multi_modal_icu_los_prediction"
   git status                      # must say "nothing to commit, working tree clean"
   git remote add origin https://github.com/<your-username>/fate-icu.git
   git push -u origin main         # a browser window asks you to sign in to GitHub
   git ls-remote origin            # the first line must show the id that "git rev-parse HEAD" prints
   ```

2. **Checksum lists**, so a damaged copy is detected before this laptop is wiped
   (the second command takes about a minute; add `--exclude` words only as shown):

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT = "C:\venvs\fate-icu"
   uv run --no-sync python -m fate_icu.utils.checksums make D:\fate-data --exclude "rag_dryrun_*"
   uv run --no-sync python -m fate_icu.utils.checksums make C:\mimic
   uv run --no-sync python -m fate_icu.utils.checksums make "$env:USERPROFILE\.cache\huggingface\hub" --exclude ".locks"
   ```

   Each command writes `SHA256SUMS.txt` into the folder it lists.

3. **A second copy of the code and the database**, as single files (Claude made these on
   2026-11-15 in `D:\fate-data\transfer\`; make them again right before you leave, because the
   bundle does not contain later commits):

   ```powershell
   New-Item -ItemType Directory -Force D:\fate-data\transfer | Out-Null
   git bundle create D:\fate-data\transfer\fate-icu.bundle --all
   $env:PGUSER = "postgres"        # pg_dump asks for the password
   pg_dump -Fc -d mimiciv_demo -f D:\fate-data\transfer\mimiciv_demo.dump
   ```

4. **Copy to an external drive** (or any storage you trust), keeping the folder names:

   - `D:\fate-data\` (4.8 GB; `rag_dryrun_*` may be left out)
   - `C:\mimic\`
   - `C:\Users\LENOVO\.cache\huggingface\hub\` without its `.locks` folder
   - your `.env`, only if you are carrying it by hand on a drive you control. Otherwise write
     down the values instead of copying the file.

### B. On the new computer (Windows; allow 1.5 hours)

1. **Install the tools**, then open a new terminal:

   ```powershell
   winget install --id Git.Git -e
   winget install --id astral-sh.uv -e
   ```

   PostgreSQL: the EDB installer from <https://www.postgresql.org/download/windows/> with
   "Command Line Tools" ticked, version 18 or newer if you will restore the dump, 14 or newer
   if you will rebuild. Note the `postgres` password. Add its `bin` folder to PATH as in
   section 1. Install the Claude desktop app and sign in.

2. **Get the code** into a short path that is not in OneDrive (long paths and OneDrive both
   caused trouble on this laptop):

   ```powershell
   git clone https://github.com/<your-username>/fate-icu.git C:\fate-icu
   # without GitHub, from the bundle: git clone D:\fate-data\transfer\fate-icu.bundle C:\fate-icu
   Set-Location C:\fate-icu
   ```

3. **Python environment.** `uv sync` installs Python 3.11 and every package at the exact
   version in `uv.lock` (about 2 GB, 5 to 15 minutes):

   ```powershell
   [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", "C:\venvs\fate-icu", "User")
   $env:UV_PROJECT_ENVIRONMENT = "C:\venvs\fate-icu"
   $env:UV_HTTP_TIMEOUT = "300"
   uv sync
   uv run pre-commit install
   ```

4. **Put the data back** where it was (`D:\fate-data\`, `C:\mimic\`, and the Hugging Face
   models into `$env:USERPROFILE\.cache\huggingface\hub\`; another drive letter is fine if you
   set `FATE_RAG_DIR` to match), then verify each copy. Every command must end with `"ok": true`:

   ```powershell
   uv run python -m fate_icu.utils.checksums check D:\fate-data
   uv run python -m fate_icu.utils.checksums check C:\mimic
   uv run python -m fate_icu.utils.checksums check "$env:USERPROFILE\.cache\huggingface\hub"
   ```

5. **`.env`**: `Copy-Item .env.example .env`, then `notepad .env` and fill in (leave
   `MIMIC_RAW_DIR` commented out):

   ```
   FATE_MACHINE=laptop
   FATE_DB_URL=postgresql+psycopg://postgres:<password>@localhost:5432/mimiciv_demo
   FATE_RAG_DIR=D:\fate-data\rag
   MIMIC_DEMO_DIR=C:/mimic/mimic-iv-clinical-database-demo-2.2
   ICD9_GEM_PATH=C:/mimic/icd9to10gem/2018_I9gem.txt
   NCBI_EMAIL=<your email, only needed to download from PubMed again>
   ```

6. **The database**, either way (the check at the end of section 3 must show 140 ICU stays):

   ```powershell
   $env:PGUSER = "postgres"
   createdb mimiciv_demo
   pg_restore -d mimiciv_demo --no-owner D:\fate-data\transfer\mimiciv_demo.dump   # 2 minutes
   ```

   or rebuild it from `C:\mimic` with the commands of section 3 (20 minutes).

7. **Check that everything works**:

   ```powershell
   $env:HF_HUB_OFFLINE = "1"     # proves the copied MedCPT models are used
   uv run pytest                                                          # expect 0 failures, 2 skipped
   uv run python -m fate_icu.tabular.run --config configs/tabular/smoke.yaml
   uv run python -m fate_icu.rag.fetch --config configs/rag/full.yaml --source guidelines
   uv run python -m fate_icu.rag.build_all --config configs/rag/smoke.yaml   # 7 minutes, real MedCPT on the CPU
   ```

   The fetch command must list the four guideline PDFs as present.

8. **Continue with Claude Code.** Open `C:\fate-icu` in the Claude desktop app. `CLAUDE.md`
   loads on its own; the conversation history does not move, so the state lives in the docs.
   First message: "Read docs/STATUS.md and docs/decisions.md, then continue Stage 5 where it
   stopped ('Resume here')."

### C. Before wiping this laptop

1. The new computer passed step 7, and `git ls-remote origin` showed your latest commit.
2. Anything else you want is copied. This guide covers only this project; your other folders
   (for example `D:\project`, `D:\Research paper`, `D:\Work`, `C:\dev` and `Downloads`) were
   not looked at.
3. Sign out and revoke: GitHub (Windows Credential Manager, entries starting `git:`), Claude,
   Kaggle, PhysioNet, OneDrive (Settings > Account > Unlink this PC) and your Microsoft
   account's device list.
4. Keep the external-drive copy for a week or two, until the new computer has run the
   pipeline once.
5. Settings > System > Recovery > Reset this PC > Remove everything, and under "Change
   settings" turn on "Clean data" and "Delete files from all drives".

### Packages (nothing to install by hand except the tools in B1)

System tools: Git for Windows (its `usr\bin` holds the `gzip` that mimic-code's loader
needs), uv (it installs Python 3.11 itself), PostgreSQL with `psql`, `createdb`, `pg_dump`
and `pg_restore` on PATH. Only for the lab, later: wget.

Python packages come from `pyproject.toml` and `uv.lock` through `uv sync`. Versions on this
laptop:

```
numpy 2.3.5, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, xgboost 3.2.0, joblib 1.6.0,
pyarrow 26.0.0, pyyaml 6.0.3, pydantic 2.14.2, python-dotenv 1.2.4, sqlalchemy 2.1.6,
psycopg[binary] 3.3.6
torch 2.15.0, torchvision 0.29.2, torchxrayvision 1.5.7, scikit-image 0.26.0, pillow 13.0.0
bm25s 0.3.16, PyStemmer 3.1.0, faiss-cpu 1.15.1, transformers 5.22.2, huggingface-hub 2.4.0,
pypdf 7.0.0
development: pytest 9.2.0, ruff 0.17.3, pre-commit 4.7.0
```

On macOS or Linux the steps are the same with: Git, uv and PostgreSQL from `brew` or `apt`;
`UV_PROJECT_ENVIRONMENT=$HOME/venvs/fate-icu`; POSIX paths in `.env` (for example
`FATE_RAG_DIR=/data/fate-data/rag`); the Hugging Face cache at `~/.cache/huggingface/hub`;
and no `gzip` or PATH steps, since both come with the system.
