WITH age_counts AS (
    SELECT 
        age,
        COUNT(*) AS encounter_count,
        SUM(
            CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END
        ) AS early_readmission_count
    FROM encounters_staging
    GROUP BY age
)
SELECT
    age,
    encounter_count,
    early_readmission_count,
    100.0 * early_readmission_count / encounter_count AS early_readmission_percentage
FROM age_counts
ORDER BY age;