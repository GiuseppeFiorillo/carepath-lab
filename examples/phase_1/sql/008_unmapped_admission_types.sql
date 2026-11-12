SELECT COUNT(*) AS encounters_without_mapping
FROM encounters_staging AS encounters
LEFT JOIN admission_type_mapping AS mapping
    ON encounters.admission_type_id = mapping.admission_type_id
WHERE mapping.admission_type_id IS NULL