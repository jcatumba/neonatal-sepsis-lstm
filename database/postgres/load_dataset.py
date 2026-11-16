"""Verify and load the official PICDB v1.1.0 CSV tables into local PostgreSQL."""

import argparse
import csv
import hashlib
from pathlib import Path
import re
import sys

import psycopg


EXPECTED_TABLES = {
    "ADMISSIONS", "CHARTEVENTS", "DIAGNOSES_ICD", "D_ICD_DIAGNOSES",
    "D_ITEMS", "D_LABITEMS", "EMR_SYMPTOMS", "ICUSTAYS", "INPUTEVENTS",
    "LABEVENTS", "MICROBIOLOGYEVENTS", "OR_EXAM_REPORTS", "OUTPUTEVENTS",
    "PATIENTS", "PRESCRIPTIONS", "SURGERY_INFO", "SURGERY_VITAL_SIGNS",
}
SCHEMA = "picdb_v1_1_0"
SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def parse_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}.")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def verify_sources(root: Path) -> dict[str, tuple[Path, str, int]]:
    root = root.resolve()
    checksum_file = root / "SHA256SUMS.txt"
    source_dir = root / "V1.1.0"
    if not checksum_file.is_file() or not source_dir.is_dir():
        raise FileNotFoundError("Expected SHA256SUMS.txt and V1.1.0/ under --data-dir.")
    checksums = {}
    for line in checksum_file.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, relative_path = parts
        checksums[relative_path.removeprefix("*")] = digest.lower()

    files = {path.stem: path for path in source_dir.glob("*.csv")}
    if set(files) != EXPECTED_TABLES:
        missing, extra = sorted(EXPECTED_TABLES - files.keys()), sorted(files.keys() - EXPECTED_TABLES)
        raise ValueError(f"Unexpected v1.1.0 table set; missing={missing}, extra={extra}.")

    verified = {}
    for table_name, path in sorted(files.items()):
        relative = path.relative_to(root).as_posix()
        expected = checksums.get(relative)
        if expected is None:
            raise ValueError(f"No official checksum listed for {relative}.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}.")
        verified[table_name] = (path, actual, path.stat().st_size)
    print(f"Verified {len(verified)} v1.1.0 source files against SHA256SUMS.txt.")
    return verified


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def connect(env_file: Path) -> psycopg.Connection:
    env = parse_env_file(env_file)
    return psycopg.connect(
        host="127.0.0.1",
        port=int(env.get("PICDB_POSTGRES_PORT", "55432")),
        dbname=env.get("PICDB_POSTGRES_DB", "picdb"),
        user=env.get("PICDB_POSTGRES_USER", "picdb"),
        password=env["PICDB_POSTGRES_PASSWORD"],
        connect_timeout=10,
        autocommit=True,
    )


def initialize_schema(connection: psycopg.Connection) -> None:
    for statement in SCHEMA_FILE.read_text().split(";"):
        if statement.strip():
            connection.execute(statement)


def existing_state(connection: psycopg.Connection) -> tuple[set[str], dict[str, tuple[str, str]]]:
    tables = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
          AND table_name <> 'dataset_file_manifest'
        ORDER BY table_name;
        """,
        (SCHEMA,),
    ).fetchall()
    manifest_rows = connection.execute(
        f"""
        SELECT table_name, trim(source_sha256), source_file
        FROM {SCHEMA}.dataset_file_manifest
        ORDER BY table_name;
        """
    ).fetchall()
    tables = {row[0] for row in tables}
    manifest = {
        table: (digest, source_file)
        for table, digest, source_file in manifest_rows
    }
    return tables, manifest


def import_table(connection: psycopg.Connection, table_name: str, source: Path,
                 digest: str, size: int) -> None:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Invalid or duplicate column names in {source.name}.")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", column) for column in columns):
        raise ValueError(f"Unsupported source column name in {source.name}.")

    relation = f"{SCHEMA}.{table_name.lower()}"
    column_sql = ", ".join(quote_identifier(column.lower()) for column in columns)
    definitions = ", ".join(
        f"{quote_identifier(column.lower())} text" for column in columns
    )
    with connection.transaction():
        connection.execute(f"CREATE TABLE {relation} ({definitions})")
        with connection.cursor().copy(
            f"COPY {relation} ({column_sql}) FROM STDIN "
            "WITH (FORMAT csv, HEADER true, NULL '')"
        ) as copy:
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    copy.write(block)
        connection.execute(
            f"""
        INSERT INTO {SCHEMA}.dataset_file_manifest
            (table_name, source_file, source_sha256, source_bytes, imported_rows)
        SELECT %s, %s, %s, %s, count(*) FROM {relation}
            """,
            (table_name.lower(), f"V1.1.0/{source.name}", digest, size),
        )
    print(f"Imported {table_name}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="PICDB release root containing V1.1.0/ and SHA256SUMS.txt")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    env_file = args.env_file.resolve()
    if not env_file.is_file():
        raise FileNotFoundError(f"Create {env_file} from .env.example and set a unique password.")
    env = parse_env_file(env_file)
    configured_root = Path(env.get("PICDB_DATA_DIR", "")).expanduser().resolve()
    data_root = args.data_dir.expanduser().resolve()
    if configured_root != data_root:
        raise ValueError("--data-dir must match PICDB_DATA_DIR in the Compose .env file.")

    verified = verify_sources(data_root)
    with connect(env_file) as connection:
        initialize_schema(connection)
        existing_tables, imported = existing_state(connection)
        for table_name, (source, digest, size) in verified.items():
            normalized = table_name.lower()
            existing = imported.get(normalized)
            if normalized in existing_tables:
                if existing != (digest, f"V1.1.0/{source.name}"):
                    raise RuntimeError(
                        f"{table_name} exists without a matching verified load manifest; "
                        "refusing to overwrite data."
                    )
                print(f"Already verified: {table_name}.")
                continue
            if existing is not None:
                raise RuntimeError(f"Manifest exists but raw table {table_name} is missing.")
            import_table(connection, table_name, source, digest, size)

    print("All expected raw tables are present and linked to verified source checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
