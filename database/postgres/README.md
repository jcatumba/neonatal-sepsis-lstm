# Local PICDB PostgreSQL workflow

This imports the PICDB v1.1.0 CSV release into a new, local-only PostgreSQL
container, verifies the source checksums, and provides aggregate integrity and
paper-aligned profiling queries. It does not import data into Google Drive or
Colab, alter source files, generate a patient-level extract, or train a model.

## Requirements

- Docker Engine with Docker Compose v2.
- Python 3.10+ and `psycopg[binary]` (see `requirements.txt`).
- The authorized PICDB v1.1.0 directory and the supplied `SHA256SUMS.txt`.
- A unique local database password. Keep the `.env` file private.

The service uses the existing `postgres:16` image, stores its database in the
named Docker volume `picdb-repro_picdb_postgres_data`, and exposes PostgreSQL
only on `127.0.0.1:55432`. It does not mount the dataset, reuse, or modify other
PostgreSQL containers. Do not publish this port or container to an unapproved
network.

## Create and load

From the project root:

```bash
cp database/postgres/.env.example database/postgres/.env
```

Edit the ignored `.env` file. Set `PICDB_DATA_DIR` to the absolute path of the
directory containing `V1.1.0/` and `SHA256SUMS.txt`, and replace the example
password with a unique strong local password. Do not commit or share `.env`.

Then start and load:

```bash
python -m pip install -r database/postgres/requirements.txt

docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml up -d postgres

python database/postgres/load_dataset.py \
  --data-dir "/absolute/path/to/paediatric-intensive-care-database-1.1.0"
```

The importer streams CSV bytes from the local Python process to PostgreSQL over
the loopback-only port. The container does not get access to the dataset path,
and source directory/file permissions and security labels are not changed.

The loader verifies all 17 v1.1.0 CSV hashes against `SHA256SUMS.txt` before
import, creates tables in the isolated `picdb_v1_1_0` schema, imports columns
as text, and records the source path, SHA-256, byte size and row count. Import
transactions are per table; a failed table is rolled back without deleting
successfully imported tables. A rerun verifies and skips tables already
imported from identical source hashes. Existing tables without a matching
manifest are never overwritten. The host importer reads the source files and
streams their bytes to PostgreSQL over the loopback connection; the container
does not mount or access the dataset directory.

Empty CSV fields load as SQL `NULL`. Source files remain unchanged and retain
their official checksums; the PostgreSQL representation is for analysis, not a
byte-for-byte CSV archive.

## Validate and profile

Run aggregate SQL only; the provided queries do not return patient IDs:

```bash
docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml exec -T postgres \
  psql -U picdb -d picdb -v ON_ERROR_STOP=1 \
  < database/postgres/sql/validate.sql

docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml exec -T postgres \
  psql -U picdb -d picdb -v ON_ERROR_STOP=1 \
  < database/postgres/sql/article_analysis.sql

docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml exec -T postgres \
  psql -U picdb -d picdb -v ON_ERROR_STOP=1 \
  < database/postgres/sql/landmark_profile.sql

docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml exec -T postgres \
  psql -U picdb -d picdb -v ON_ERROR_STOP=1 \
  < database/postgres/sql/biomarker_coverage.sql
```

Inspect row counts, source hashes, duplicate IDs, core join orphans, outcome
disagreements, NICU cohort counts, candidate biomarker item mappings/coverage,
and timing relative to death. Compare aggregate counts with the article's
Tables 2-4 before defining a modelling extract. The queries intentionally show
the difference between NICU admissions and NICU ICU stays; that cohort choice
is a reproduction decision, not an automatic assumption.
The landmark profile explores 6-, 12-, and 24-hour observation windows from
hospital admission. It is an aggregate availability check, not a training
extract or a chosen clinical prediction protocol. In particular, a nonempty
`VALUENUM` field is not necessarily a validated numeric observation, and
multiple events do not establish that a usable 10-step sequence exists.
The biomarker coverage query explores ten equal bins in the selected first
24-hour window for lactate, total bilirubin, and creatinine. It counts only
numeric-text values in the source units listed in the query, does not impute
values, and reports aggregates rather than patient-level data.

## Build the local candidate mortality dataset

The user-selected candidate protocol is documented in
[`Data and code review and preparation.md`](../../Data%20and%20code%20review%20and%20preparation.md).
With the service running, the verified raw tables loaded, and the dependencies
from both `requirements-colab.txt` and `database/postgres/requirements.txt`
installed in the Python environment, run from the project root:

```bash
python -m database.postgres.build_mortality_dataset
```

By default this creates the new, Git-ignored `outputs/mortality-dataset/`
directory with `dataset.npz` and `summary.json`. The directory is created with
mode `700` and the files with mode `600`; an existing output directory is
never overwritten. Choose another new directory with `--output-dir`. An
output path inside the repository must be under the ignored `outputs/` tree.
Restrictive permissions do not replace institutional access controls: the NPZ
contains patient and admission identifiers and derived clinical data. Keep it
private, never commit it, and do not upload it without explicit authorization.

Each split has arrays `train_x`, `validation_x`, or `test_x` of shape
`(admissions, 10, 6)` and corresponding `_y`, `_patient_id`, `_admission_id`,
and `_birth_after_admission` arrays. The first three channels are scaled
lactate, total bilirubin, and creatinine; the last three indicate whether each
marker was observed in that bin, before imputation. Files can be read with
`np.load(path, allow_pickle=False)`. All admissions from a patient stay in one
split. The 70/15/15 patient split is stratified by whether the patient has any
included in-hospital death and uses seed 42 by default.

The last valid observation in each 2.4-hour bin is selected, with the largest
numeric lab `ROW_ID` breaking timestamp ties. Candidate events with missing or
incorrect units or nonnumeric/nonfinite values are excluded and counted.
Within each admission, missing bins are forward-filled; initial gaps use
per-marker medians of observed training bins. The min-max scaler is fitted
only on the completed training sequences; validation and test values are not
clipped. The summary records the medians, scaler bounds, source import
manifest, versions, exclusions and aggregate split counts. No physiological
range filtering is applied. These preprocessing choices are approximations,
not established details of the article.

The existing Colab sepsis-demo notebook does **not** consume this format or
implement this mortality experiment. The builder does not train a model; the
labels are retained for stratification and later outcome evaluation.

Connect locally with a PostgreSQL client using host `127.0.0.1`, port `55432`,
database/user from `.env`, and the password set there. The database is
persistent across container restarts. The supplied query commands above read
SQL files from the host and stream them to the container. Stop the service
without deleting data:

```bash
docker compose --project-name picdb-repro \
  --env-file database/postgres/.env \
  -f database/postgres/compose.yaml stop postgres
```

## Data-use note

PICDB ships with the PhysioNet Credentialed Health Data License. The local
workflow keeps records on the local machine and emits aggregate analysis.
De-identification is not a substitute for credentialing, security, or approval.
Before moving any data to Drive, Colab, or another cloud service, verify the
license, institutional approval, account controls, retention and sharing
settings. Do not put row-level data, patient identifiers, credentials, or
patient-level query results into this repository.
