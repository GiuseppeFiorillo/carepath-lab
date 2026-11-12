SELECT
    encounters.admission_type_id,
    mapping.description,
    COUNT(*) AS encounter_count
FROM encounters_staging AS encounters
LEFT JOIN admission_type_mapping AS mapping
    ON encounters.admission_type_id = mapping.admission_type_id
GROUP BY
    encounters.admission_type_id,
    mapping.description
ORDER BY encounters.admission_type_id