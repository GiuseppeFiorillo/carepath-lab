SELECT
    readmitted,
    COUNT(*) AS encounter_count
FROM encounters_staging
GROUP BY readmitted
ORDER BY readmitted;