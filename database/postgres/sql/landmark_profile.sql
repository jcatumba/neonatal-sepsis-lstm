\pset pager off
\echo '== Exploratory NICU admission landmarks; aggregate counts only =='
WITH nicu AS (
    SELECT row_id, subject_id, hadm_id, hospital_expire_flag,
           NULLIF(admittime, '')::timestamp AS admission_time,
           NULLIF(dischtime, '')::timestamp AS discharge_time,
           NULLIF(deathtime, '')::timestamp AS death_time
    FROM picdb_v1_1_0.admissions
    WHERE admission_department = 'NICU'
), landmarks AS (
    SELECT n.*, h.hours, n.admission_time + h.hours * interval '1 hour' AS landmark
    FROM nicu AS n
    CROSS JOIN (VALUES (6), (12), (24)) AS h(hours)
), eligible AS (
    SELECT *
    FROM landmarks
    WHERE admission_time IS NOT NULL
      AND discharge_time >= landmark
      AND (death_time IS NULL OR death_time > landmark)
), event_counts AS (
    SELECT e.row_id, e.hours,
           count(l.itemid) AS events,
           count(l.itemid) FILTER (WHERE NULLIF(l.valuenum, '') IS NOT NULL) AS events_with_valuenum,
           bool_or(l.itemid = '5227') AS lactate,
           bool_or(l.itemid IN ('5075', '5255')) AS total_bilirubin,
           bool_or(l.itemid IN ('5032', '5041')) AS creatinine
    FROM eligible AS e
    LEFT JOIN picdb_v1_1_0.labevents AS l
      ON l.subject_id = e.subject_id AND l.hadm_id = e.hadm_id
     AND NULLIF(l.charttime, '')::timestamp >= e.admission_time
     AND NULLIF(l.charttime, '')::timestamp < e.landmark
    GROUP BY e.row_id, e.hours
)
SELECT e.hours,
       count(*) AS eligible_admissions,
       count(DISTINCT e.subject_id) AS eligible_subjects,
       count(*) FILTER (WHERE e.hospital_expire_flag = '1') AS eligible_death_admissions,
       count(*) FILTER (WHERE e.hospital_expire_flag = '1' AND e.death_time IS NULL)
           AS death_admissions_without_death_time,
       count(*) FILTER (WHERE c.events > 0) AS admissions_with_any_lab,
       count(*) FILTER (WHERE c.events >= 10) AS admissions_with_at_least_10_labs,
       count(*) FILTER (WHERE c.events_with_valuenum >= 10) AS admissions_with_at_least_10_valuenum_fields,
       count(*) FILTER (WHERE c.lactate AND c.total_bilirubin AND c.creatinine)
           AS admissions_with_three_candidate_markers,
       count(*) FILTER (WHERE e.hospital_expire_flag = '1' AND c.events >= 10)
           AS death_admissions_with_at_least_10_labs
FROM eligible AS e
JOIN event_counts AS c USING (row_id, hours)
GROUP BY e.hours
ORDER BY e.hours;
