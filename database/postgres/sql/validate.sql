\pset pager off
\echo '== Expected source table and manifest counts =='
SELECT
    (SELECT count(*) FROM information_schema.tables
     WHERE table_schema = 'picdb_v1_1_0' AND table_type = 'BASE TABLE'
       AND table_name <> 'dataset_file_manifest') AS raw_tables,
    (SELECT count(*) FROM picdb_v1_1_0.dataset_file_manifest) AS manifest_rows,
    (SELECT count(*) FROM picdb_v1_1_0.dataset_file_manifest
     WHERE source_sha256 ~ '^[0-9a-f]{64}$') AS valid_sha256_rows;

\echo '== Row counts recorded at import =='
SELECT table_name, imported_rows
FROM picdb_v1_1_0.dataset_file_manifest
ORDER BY table_name;

\echo '== Duplicate source row IDs (expected zero where row_id exists) =='
SELECT 'admissions' AS table_name, count(*) - count(DISTINCT row_id) AS duplicate_row_ids FROM picdb_v1_1_0.admissions
UNION ALL SELECT 'chartevents', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.chartevents
UNION ALL SELECT 'diagnoses_icd', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.diagnoses_icd
UNION ALL SELECT 'd_icd_diagnoses', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.d_icd_diagnoses
UNION ALL SELECT 'd_items', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.d_items
UNION ALL SELECT 'd_labitems', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.d_labitems
UNION ALL SELECT 'emr_symptoms', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.emr_symptoms
UNION ALL SELECT 'icustays', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.icustays
UNION ALL SELECT 'inputevents', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.inputevents
UNION ALL SELECT 'labevents', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.labevents
UNION ALL SELECT 'microbiologyevents', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.microbiologyevents
UNION ALL SELECT 'or_exam_reports', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.or_exam_reports
UNION ALL SELECT 'outputevents', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.outputevents
UNION ALL SELECT 'patients', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.patients
UNION ALL SELECT 'prescriptions', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.prescriptions
UNION ALL SELECT 'surgery_info', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.surgery_info
UNION ALL SELECT 'surgery_vital_signs', count(*) - count(DISTINCT row_id) FROM picdb_v1_1_0.surgery_vital_signs
ORDER BY table_name;

\echo '== Referential integrity: orphan counts for core analysis joins =='
SELECT 'admissions.subject_id -> patients' AS relationship, count(*) AS orphan_rows
FROM picdb_v1_1_0.admissions a LEFT JOIN picdb_v1_1_0.patients p USING (subject_id)
WHERE p.subject_id IS NULL
UNION ALL
SELECT 'icustays.subject_id -> patients', count(*)
FROM picdb_v1_1_0.icustays i LEFT JOIN picdb_v1_1_0.patients p USING (subject_id)
WHERE p.subject_id IS NULL
UNION ALL
SELECT 'icustays.hadm_id -> admissions', count(*)
FROM picdb_v1_1_0.icustays i LEFT JOIN picdb_v1_1_0.admissions a USING (hadm_id)
WHERE a.hadm_id IS NULL
UNION ALL
SELECT 'labevents.subject_id -> patients', count(*)
FROM picdb_v1_1_0.labevents l LEFT JOIN picdb_v1_1_0.patients p USING (subject_id)
WHERE p.subject_id IS NULL
UNION ALL
SELECT 'labevents.hadm_id -> admissions', count(*)
FROM picdb_v1_1_0.labevents l LEFT JOIN picdb_v1_1_0.admissions a USING (hadm_id)
WHERE a.hadm_id IS NULL
UNION ALL
SELECT 'labevents.itemid -> d_labitems', count(*)
FROM picdb_v1_1_0.labevents l LEFT JOIN picdb_v1_1_0.d_labitems d USING (itemid)
WHERE d.itemid IS NULL
ORDER BY relationship;

\echo '== Core tables analyzed for query planning =='
ANALYZE picdb_v1_1_0.patients;
ANALYZE picdb_v1_1_0.admissions;
ANALYZE picdb_v1_1_0.icustays;
ANALYZE picdb_v1_1_0.d_labitems;
ANALYZE picdb_v1_1_0.labevents;
