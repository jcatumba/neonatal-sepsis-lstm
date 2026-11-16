CREATE SCHEMA IF NOT EXISTS picdb_v1_1_0;

CREATE TABLE IF NOT EXISTS picdb_v1_1_0.dataset_file_manifest (
    table_name text PRIMARY KEY,
    source_file text NOT NULL UNIQUE,
    source_sha256 char(64) NOT NULL,
    source_bytes bigint NOT NULL CHECK (source_bytes >= 0),
    imported_rows bigint NOT NULL CHECK (imported_rows >= 0),
    loaded_at timestamptz NOT NULL DEFAULT now()
);
