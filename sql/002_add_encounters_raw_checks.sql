-- Run once, after 001_create_encounters_raw.sql.
-- Enforce the minimal ingestion contract without converting source values.
-- BTRIM checks empty strings and strings containing only ordinary spaces;
-- it does not change the stored identifiers or validate their full format.
ALTER TABLE encounters_raw
    ADD CONSTRAINT encounters_raw_encounter_id_not_blank
        CHECK (BTRIM(encounter_id) <> ''),
    ADD CONSTRAINT encounters_raw_patient_nbr_not_blank
        CHECK (BTRIM(patient_nbr) <> ''),
    ADD CONSTRAINT encounters_raw_readmitted_allowed
        CHECK (readmitted IN ('<30', '>30', 'NO'));
