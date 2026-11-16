\pset pager off
\pset null '(NULL)'
\echo '== Imported source table manifest (no patient-level rows) =='
SELECT table_name, source_file, trim(source_sha256) AS sha256,
       source_bytes, imported_rows
FROM picdb_v1_1_0.dataset_file_manifest
ORDER BY table_name;

\echo '== Global patient outcome counts =='
SELECT expire_flag,
       count(*) AS patients,
       count(*) FILTER (WHERE NULLIF(dod, '') IS NOT NULL) AS patients_with_death_time
FROM picdb_v1_1_0.patients
GROUP BY expire_flag
ORDER BY expire_flag;

\echo '== Admission counts by source department and admission death flag =='
SELECT admission_department,
       hospital_expire_flag,
       count(*) AS admissions,
       count(DISTINCT subject_id) AS distinct_patients
FROM picdb_v1_1_0.admissions
GROUP BY admission_department, hospital_expire_flag
ORDER BY admission_department, hospital_expire_flag;

\echo '== Article Table 2 claims compared with source admission counts =='
WITH paper_claims(article_department, source_department, live_claim, deceased_claim) AS (
    VALUES
        ('Cardiac surgery department', 'Cardiac surgery department', 2238, 2),
        ('General ICU (possible CICU label mismatch)', 'CICU', 43, 50),
        ('Infectious diseases department', 'Infectious diseases department', 50, 1),
        ('PICU', 'PICU', 1246, 205),
        ('SICU', 'SICU', 114, 57),
        ('NICU', 'NICU', 2081, 239)
), local_counts AS (
    SELECT admission_department,
           count(*) FILTER (WHERE hospital_expire_flag = '0') AS live_admissions,
           count(*) FILTER (WHERE hospital_expire_flag = '1') AS deceased_admissions
    FROM picdb_v1_1_0.admissions
    GROUP BY admission_department
)
SELECT p.article_department,
       p.live_claim,
       p.deceased_claim,
       l.live_admissions AS local_live_admissions,
       l.deceased_admissions AS local_deceased_admissions,
       (p.live_claim = l.live_admissions
        AND p.deceased_claim = l.deceased_admissions) AS exact_count_match
FROM paper_claims AS p
LEFT JOIN local_counts AS l ON l.admission_department = p.source_department
UNION ALL
SELECT 'General ICU (literal PICDB label)', NULL, NULL,
       live_admissions, deceased_admissions, NULL
FROM local_counts
WHERE admission_department = 'General ICU'
ORDER BY article_department;

\echo '== ICU stay counts by care unit and patient outcome =='
SELECT first_careunit,
       p.expire_flag AS patient_expire_flag,
       count(*) AS icu_stays,
       count(DISTINCT i.subject_id) AS distinct_patients
FROM picdb_v1_1_0.icustays AS i
JOIN picdb_v1_1_0.patients AS p USING (subject_id)
GROUP BY first_careunit, p.expire_flag
ORDER BY first_careunit, p.expire_flag;

\echo '== NICU admission-department vs ICU-stay cohort comparison =='
WITH admission_cohort AS (
    SELECT DISTINCT a.subject_id
    FROM picdb_v1_1_0.admissions AS a
    WHERE upper(a.admission_department) LIKE '%NICU%'
), stay_cohort AS (
    SELECT DISTINCT i.subject_id
    FROM picdb_v1_1_0.icustays AS i
    WHERE upper(i.first_careunit) LIKE '%NICU%'
       OR upper(i.last_careunit) LIKE '%NICU%'
), union_cohort AS (
    SELECT subject_id FROM admission_cohort
    UNION
    SELECT subject_id FROM stay_cohort
)
SELECT 'admission department NICU' AS cohort,
       count(*) AS patients,
       count(*) FILTER (WHERE p.expire_flag = '1') AS deceased_by_patient_flag
FROM admission_cohort AS c JOIN picdb_v1_1_0.patients AS p USING (subject_id)
UNION ALL
SELECT 'ICU stay unit NICU',
       count(*),
       count(*) FILTER (WHERE p.expire_flag = '1')
FROM stay_cohort AS c JOIN picdb_v1_1_0.patients AS p USING (subject_id)
UNION ALL
SELECT 'union of NICU definitions',
       count(*),
       count(*) FILTER (WHERE p.expire_flag = '1')
FROM union_cohort AS c JOIN picdb_v1_1_0.patients AS p USING (subject_id);

\echo '== Mortality signal disagreements within admission-department NICU cohort =='
WITH nicu AS (
    SELECT a.subject_id,
           bool_or(a.hospital_expire_flag = '1') AS admission_death_flag
    FROM picdb_v1_1_0.admissions AS a
    WHERE upper(a.admission_department) LIKE '%NICU%'
    GROUP BY a.subject_id
)
SELECT count(*) AS patients,
       count(*) FILTER (WHERE p.expire_flag = '1') AS deceased_patient_flag,
       count(*) FILTER (WHERE n.admission_death_flag) AS deceased_admission_flag,
       count(*) FILTER (WHERE NULLIF(p.dod, '') IS NOT NULL) AS with_patient_death_time,
       count(*) FILTER (
           WHERE (p.expire_flag = '1') <> n.admission_death_flag
       ) AS patient_admission_flag_disagreements
FROM nicu AS n
JOIN picdb_v1_1_0.patients AS p USING (subject_id);

\echo '== NICU age-at-death distribution, patient level =='
WITH nicu AS (
    SELECT DISTINCT a.subject_id
    FROM picdb_v1_1_0.admissions AS a
    WHERE upper(a.admission_department) LIKE '%NICU%'
), deaths AS (
    SELECT DISTINCT p.subject_id,
           CASE WHEN NULLIF(p.dod, '') IS NOT NULL AND NULLIF(p.dob, '') IS NOT NULL
                THEN extract(epoch FROM (
                    NULLIF(p.dod, '')::timestamp - NULLIF(p.dob, '')::timestamp
                )) / 86400.0
           END AS age_days
    FROM nicu AS n
    JOIN picdb_v1_1_0.patients AS p USING (subject_id)
    WHERE p.expire_flag = '1'
)
SELECT count(*) AS deceased_nicu_patients,
       count(*) FILTER (WHERE age_days BETWEEN 0 AND 210) AS age_0_210_days,
       count(*) FILTER (WHERE age_days < 0 OR age_days > 210) AS outside_0_210_days,
       count(*) FILTER (WHERE age_days IS NULL) AS missing_birth_or_death_time,
       min(age_days) AS minimum_age_days,
       max(age_days) AS maximum_age_days
FROM deaths;

\echo '== Candidate biomarkers and observed laboratory availability =='
SELECT l.itemid,
       d.label,
       d.fluid,
       d.category,
       count(*) AS lab_events,
       count(*) FILTER (WHERE NULLIF(l.valuenum, '') IS NOT NULL) AS numeric_events,
       count(DISTINCT l.subject_id) AS distinct_patients,
       count(*) FILTER (WHERE NULLIF(l.valueuom, '') IS NOT NULL) AS events_with_units,
       count(*) FILTER (WHERE NULLIF(l.flag, '') IS NOT NULL) AS events_with_flags,
       min(NULLIF(l.charttime, '')::timestamp) AS first_event,
       max(NULLIF(l.charttime, '')::timestamp) AS last_event
FROM picdb_v1_1_0.d_labitems AS d
JOIN picdb_v1_1_0.labevents AS l USING (itemid)
WHERE lower(d.label) ~ '(lactate|bilirubin|creatinine|c.?reactive protein|procalcitonin|white blood cell|platelet)'
GROUP BY l.itemid, d.label, d.fluid, d.category
ORDER BY lower(d.label), l.itemid;

\echo '== Candidate biomarker availability within admission-department NICU cohort =='
SELECT l.itemid,
       d.label,
       NULLIF(l.valueuom, '') AS source_unit,
       count(*) AS lab_events,
       count(*) FILTER (WHERE NULLIF(l.valuenum, '') IS NOT NULL) AS numeric_events,
       count(DISTINCT l.subject_id) AS distinct_patients
FROM picdb_v1_1_0.admissions AS a
JOIN picdb_v1_1_0.labevents AS l USING (subject_id, hadm_id)
JOIN picdb_v1_1_0.d_labitems AS d USING (itemid)
WHERE upper(a.admission_department) LIKE '%NICU%'
  AND lower(d.label) IN (
      'lactate', 'bilirubin, total', 'bilirubin, direct', 'creatinine',
      'c-reactive protein', 'procalcitonin', 'platelet count'
  )
GROUP BY l.itemid, d.label, NULLIF(l.valueuom, '')
ORDER BY lower(d.label), l.itemid, source_unit;

\echo '== Candidate biomarker flag distribution =='
SELECT d.label,
       NULLIF(l.flag, '') AS source_flag,
       count(*) AS events
FROM picdb_v1_1_0.d_labitems AS d
JOIN picdb_v1_1_0.labevents AS l USING (itemid)
WHERE lower(d.label) ~ '(lactate|bilirubin|creatinine|c.?reactive protein|procalcitonin|white blood cell|platelet)'
GROUP BY d.label, NULLIF(l.flag, '')
ORDER BY lower(d.label), source_flag;

\echo '== Diagnostic dictionary entries mentioning sepsis/infection =='
SELECT d.title,
       count(DISTINCT x.subject_id) AS distinct_patients_with_code,
       count(*) AS diagnosis_rows
FROM picdb_v1_1_0.diagnoses_icd AS x
JOIN picdb_v1_1_0.d_icd_diagnoses AS d USING (icd10_code_cn)
WHERE lower(d.title) ~ '(sepsis|infection)'
GROUP BY d.title
ORDER BY lower(d.title);

\echo '== Timing risk: lab events after admission death time in NICU admissions =='
WITH nicu_admissions AS (
    SELECT a.subject_id, a.hadm_id, NULLIF(a.deathtime, '')::timestamp AS death_time
    FROM picdb_v1_1_0.admissions AS a
    WHERE upper(a.admission_department) LIKE '%NICU%'
      AND NULLIF(a.deathtime, '') IS NOT NULL
)
SELECT count(*) AS lab_events_for_nicu_death_admissions,
       count(*) FILTER (
           WHERE NULLIF(l.charttime, '')::timestamp > n.death_time
       ) AS events_after_admission_death_time,
       count(DISTINCT l.subject_id) AS patients_with_lab_events,
       count(DISTINCT l.subject_id) FILTER (
           WHERE NULLIF(l.charttime, '')::timestamp > n.death_time
       ) AS patients_with_post_death_lab_events
FROM nicu_admissions AS n
JOIN picdb_v1_1_0.labevents AS l USING (subject_id, hadm_id);
