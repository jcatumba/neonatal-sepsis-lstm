\pset pager off
\echo '== Candidate 24-hour cohort and 10-bin biomarker availability (aggregate only) =='
WITH eligible AS (
    SELECT a.row_id, a.subject_id, a.hadm_id,
           NULLIF(a.admittime, '')::timestamp AS admission_time
    FROM picdb_v1_1_0.admissions AS a
    JOIN picdb_v1_1_0.patients AS p USING (subject_id)
    WHERE a.admission_department = 'NICU'
      AND NULLIF(a.admittime, '') IS NOT NULL
      AND NULLIF(a.dischtime, '')::timestamp >
          NULLIF(a.admittime, '')::timestamp + interval '24 hours'
      AND (NULLIF(a.deathtime, '') IS NULL OR NULLIF(a.deathtime, '')::timestamp >
          NULLIF(a.admittime, '')::timestamp + interval '24 hours')
), candidate_events AS (
    SELECT e.row_id,
           floor(extract(epoch FROM (
               NULLIF(l.charttime, '')::timestamp - e.admission_time
           )) / 8640)::int AS bin,
           CASE WHEN l.itemid = '5227' THEN 'lactate'
                WHEN l.itemid IN ('5075', '5255') THEN 'total_bilirubin'
                WHEN l.itemid IN ('5032', '5041') THEN 'creatinine'
           END AS marker,
           l.valuenum, l.valueuom, l.itemid
    FROM eligible AS e
    JOIN picdb_v1_1_0.labevents AS l
      ON l.subject_id = e.subject_id AND l.hadm_id = e.hadm_id
     AND NULLIF(l.charttime, '')::timestamp >= e.admission_time
     AND NULLIF(l.charttime, '')::timestamp < e.admission_time + interval '24 hours'
    WHERE l.itemid IN ('5227', '5075', '5255', '5032', '5041')
), usable AS (
    SELECT row_id, bin, marker
    FROM candidate_events
    WHERE valuenum ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
      AND ((itemid = '5227' AND valueuom = 'mmol/L')
        OR (itemid IN ('5075', '5255', '5032', '5041') AND valueuom = 'μmol/L'))
    GROUP BY row_id, bin, marker
), bins AS (
    SELECT e.row_id, b.bin,
           bool_or(u.marker = 'lactate') AS lactate,
           bool_or(u.marker = 'total_bilirubin') AS total_bilirubin,
           bool_or(u.marker = 'creatinine') AS creatinine
    FROM eligible AS e
    CROSS JOIN generate_series(0, 9) AS b(bin)
    LEFT JOIN usable AS u ON u.row_id = e.row_id AND u.bin = b.bin
    GROUP BY e.row_id, b.bin
)
SELECT bin,
       count(*) AS eligible_admissions,
       count(*) FILTER (WHERE lactate) AS lactate_observed,
       count(*) FILTER (WHERE total_bilirubin) AS total_bilirubin_observed,
       count(*) FILTER (WHERE creatinine) AS creatinine_observed,
       count(*) FILTER (WHERE lactate AND total_bilirubin AND creatinine) AS all_three_observed,
       count(*) FILTER (WHERE NOT coalesce(lactate, false)
                          AND NOT coalesce(total_bilirubin, false)
                          AND NOT coalesce(creatinine, false)) AS none_observed
FROM bins
GROUP BY bin
ORDER BY bin;
