SELECT
    age,
    COUNT(*) AS encounter_count,
    SUM(
        CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END
    ) AS early_readmission_count,
    100.0 * SUM(
        CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END
    ) / COUNT(*) AS early_readmission_percentage
FROM encounters_staging
GROUP BY age
HAVING COUNT(*) >= 1000
ORDER BY age;