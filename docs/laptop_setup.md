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

2. **PostgreSQL 16**: run the EDB installer from <https://www.postgresql.org/download/windows/>
   and keep "Command Line Tools" selected. Note the password you set for the `postgres` user.
   Add the tools to PATH for your user, then open a new terminal:

   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\PostgreSQL\16\bin", "User")
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
