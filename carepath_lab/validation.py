"""Validate source CSV structure without recoding recorded field values."""

from collections.abc import Collection, Iterator, Sequence


REQUIRED_COLUMNS = {"encounter_id", "patient_nbr", "readmitted"}


def read_header(reader: Iterator[list[str]]) -> list[str]:
    """Read a nonempty, unique header containing the required identifier/target fields."""
    columns = next(reader, None)
    if not columns:
        raise ValueError("CSV contains no header.")
    if any(not name.strip() for name in columns):
        raise ValueError("CSV contains an empty column name.")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV contains duplicate column names.")
    if not REQUIRED_COLUMNS.issubset(columns):
        raise ValueError("CSV is missing required identifier or target columns.")
    return columns


def validate_and_extract_record(
    row: Sequence[str],
    columns: Sequence[str],
    encounter_id_index: int,
    patient_nbr_index: int,
    readmitted_index: int,
    allowed_readmitted_values: Collection[str],
) -> tuple[str, str, str]:
    """Validate a row and return encounter_id, patient_nbr and readmitted value."""
    if len(row) != len(columns):
        raise ValueError("Unexpected number of fields in CSV record.")

    readmitted_value = row[readmitted_index]
    patient_nbr = row[patient_nbr_index]
    encounter_id = row[encounter_id_index]

    if readmitted_value not in allowed_readmitted_values:
        raise ValueError("Unexpected readmitted category.")
    if not encounter_id.strip():
        raise ValueError("Encounter ID is missing.")
    if not patient_nbr.strip():
        raise ValueError("Patient nbr is missing.")

    return encounter_id, patient_nbr, readmitted_value
