CREATE TABLE encounters_staging (
    encounter_id TEXT PRIMARY KEY,
    patient_nbr TEXT NOT NULL,
    readmitted TEXT NOT NULL,
    admission_type_id TEXT,
    age TEXT
);