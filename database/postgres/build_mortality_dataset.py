"""Build the selected local-only NICU mortality dataset from PostgreSQL."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn

from database.postgres.load_dataset import connect
from src.mortality_data import ITEMS, build_mortality_dataset


ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(__file__).with_name(".env")

COHORT_SQL = """
SELECT a.subject_id AS patient_id, a.hadm_id AS admission_id,
       NULLIF(a.admittime, '')::timestamp AS admission_time,
       CASE a.hospital_expire_flag WHEN '0' THEN 0 WHEN '1' THEN 1 END AS hospital_death,
       CASE WHEN NULLIF(p.dob, '') IS NOT NULL THEN
           NULLIF(p.dob, '')::timestamp > NULLIF(a.admittime, '')::timestamp
       END AS birth_after_admission,
       CASE WHEN a.hospital_expire_flag = '1' THEN
           NULLIF(a.deathtime, '') IS NULL OR
           NULLIF(a.deathtime, '')::timestamp > NULLIF(a.dischtime, '')::timestamp
       ELSE false END AS invalid_death_time
FROM picdb_v1_1_0.admissions AS a
JOIN picdb_v1_1_0.patients AS p USING (subject_id)
WHERE a.admission_department = 'NICU'
  AND NULLIF(a.admittime, '') IS NOT NULL
  AND NULLIF(a.dischtime, '')::timestamp >
      NULLIF(a.admittime, '')::timestamp + interval '24 hours'
  AND (NULLIF(a.deathtime, '') IS NULL OR NULLIF(a.deathtime, '')::timestamp >
      NULLIF(a.admittime, '')::timestamp + interval '24 hours')
"""

EVENTS_SQL = """
SELECT a.hadm_id AS admission_id, l.row_id,
       NULLIF(l.charttime, '')::timestamp AS charttime,
       l.itemid, l.valuenum, l.valueuom
FROM picdb_v1_1_0.admissions AS a
JOIN picdb_v1_1_0.labevents AS l USING (subject_id, hadm_id)
WHERE a.admission_department = 'NICU'
  AND NULLIF(a.admittime, '') IS NOT NULL
  AND NULLIF(a.dischtime, '')::timestamp >
      NULLIF(a.admittime, '')::timestamp + interval '24 hours'
  AND (NULLIF(a.deathtime, '') IS NULL OR NULLIF(a.deathtime, '')::timestamp >
      NULLIF(a.admittime, '')::timestamp + interval '24 hours')
  AND NULLIF(l.charttime, '')::timestamp >= NULLIF(a.admittime, '')::timestamp
  AND NULLIF(l.charttime, '')::timestamp <
      NULLIF(a.admittime, '')::timestamp + interval '24 hours'
  AND l.itemid = ANY(%s)
"""


def fetch_frame(connection, query: str, params: tuple = ()) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [column.name for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def private_file(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    return os.fdopen(descriptor, "wb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mortality-dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(ROOT) and not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("An output directory inside the repository must be under ignored outputs/.")
    if output_dir == ROOT / "outputs":
        raise ValueError("Choose a new private subdirectory under outputs/.")
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new private directory.")

    with connect(args.env_file.resolve()) as connection, connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        manifest = fetch_frame(connection, """
            SELECT table_name, trim(source_sha256) AS sha256, imported_rows
            FROM picdb_v1_1_0.dataset_file_manifest
            WHERE table_name IN ('admissions', 'patients', 'labevents')
            ORDER BY table_name
        """)
        if set(manifest["table_name"]) != {"admissions", "patients", "labevents"}:
            raise ValueError("Source import manifest is incomplete.")
        cohort = fetch_frame(connection, COHORT_SQL)
        if cohort["invalid_death_time"].any():
            raise ValueError("A death admission has a missing or inconsistent death time.")
        events = fetch_frame(connection, EVENTS_SQL, (list(ITEMS),))
    dataset = build_mortality_dataset(cohort, events, seed=args.seed)
    dataset.summary["source_manifest"] = manifest.to_dict(orient="records")
    dataset.summary["software_versions"] = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    with private_file(output_dir / "dataset.npz") as stream:
        np.savez_compressed(stream, **dataset.arrays)
    with private_file(output_dir / "summary.json") as stream:
        stream.write((json.dumps(dataset.summary, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(dataset.summary, indent=2, allow_nan=False))
    print(f"Private local dataset saved in {output_dir}; no model was trained.")


if __name__ == "__main__":
    main()
