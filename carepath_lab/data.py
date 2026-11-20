"""Load verified development partitions and fingerprint file contents."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from carepath_lab.cohort import select_analysis_cohort
from carepath_lab.features import attach_split_assignments
from carepath_lab.splitting import validate_split_assignments


def file_sha256(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_verified_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the verified source and return saved training/validation encounters."""
    # Check the source against the saved split provenance before reading records.
    source_path = Path("data/raw/diabetic_data.csv")
    metadata_path = Path("data/processed/split_metadata.json")
    split_assignments_path = Path("data/processed/split_assignments.csv")

    if not source_path.exists():
        raise ValueError("Source path should exist before running this script.")

    if not metadata_path.exists():
        raise ValueError("Metadata path should exist before running this script.")

    if not split_assignments_path.exists():
        raise ValueError("Split assignments path should exist before running this script.")

    source_sha256 = file_sha256(source_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if source_sha256 != metadata["source"]["sha256"]:
        raise ValueError("Source CSV SHA-256 does not match the split metadata.")

    data = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
    )

    assignments = pd.read_csv(
        split_assignments_path,
        dtype=str,
        keep_default_na=False,
    )

    # Apply the cohort rule and attach validated saved partition membership.
    cohort = select_analysis_cohort(data)

    validate_split_assignments(assignments, cohort)
    cohort_with_assignments = attach_split_assignments(cohort, assignments)

    # Construct inputs for development; leave test prediction/evaluation for later.
    train = cohort_with_assignments.loc[
        cohort_with_assignments["partition"] == "train"
    ]
    validation = cohort_with_assignments.loc[
        cohort_with_assignments["partition"] == "validation"
    ]

    return train, validation
