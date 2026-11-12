"""Create patient-disjoint split assignments without overwriting an existing CSV."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import pandas as pd

from carepath_lab.cohort import (
    DEATH_RELATED_DISCHARGE_CODES,
    select_analysis_cohort,
)
from carepath_lab.splitting import (
    split_patient_holdout,
    validate_split_assignments,
)


def main() -> None:
    # Read source strings and apply the documented cohort rule.
    source_path = Path("data/raw/diabetic_data.csv")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    data = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
    )

    cohort = select_analysis_cohort(data)

    observed_categories = set(cohort["readmitted"])
    allowed_categories = {"<30", ">30", "NO"}
    unexpected_categories = observed_categories - allowed_categories

    if unexpected_categories:
        raise ValueError(
            f"Unexpected readmitted categories: {sorted(unexpected_categories)}"
        )

    test_size = 0.2
    validation_size_within_development = 0.25
    random_state = 42

    # Reserve test patients first, then split development patients.
    development, test = split_patient_holdout(
        cohort, test_size=test_size, random_state=random_state
    )

    train, validation = split_patient_holdout(
        development,
        test_size=validation_size_within_development,
        random_state=random_state,
    )

    # Report outcome frequencies and build the assignment table.
    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print(
        f"Cohort: {len(cohort):,} encounters | "
        f"{cohort['patient_nbr'].nunique():,} patients"
    )
    print("\nPartition summary")
    print(
        f"{'Partition':<12} {'Encounters':>12} {'Patients':>10} "
        f"{'Positive':>10} {'Negative':>10} {'Prevalence':>12}"
    )
    assignment_frames = []

    for name, partition in partitions.items():
        positive_count = (partition["readmitted"] == "<30").sum()
        negative_count = len(partition) - positive_count
        prevalence = positive_count / len(partition) * 100

        print(
            f"{name:<12} {len(partition):>12,} "
            f"{partition['patient_nbr'].nunique():>10,} "
            f"{positive_count:>10,} {negative_count:>10,} "
            f"{prevalence:>11.2f}%"
        )

        partition_assignments = partition[["encounter_id", "patient_nbr"]].copy()
        partition_assignments["partition"] = name

        assignment_frames.append(partition_assignments)

    assignments = pd.concat(assignment_frames, ignore_index=True)

    # Validate before writing; exclusive creation protects existing assignments.
    validate_split_assignments(assignments, cohort)

    print("\nAssignments shape:", assignments.shape)
    print("Column names:", ", ".join(assignments.columns))

    metadata = {
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
        },
        "cohort": {
            "excluded_discharge_codes": sorted(DEATH_RELATED_DISCHARGE_CODES),
            "encounter_count": len(cohort),
        },
        "split": {
            "test_size": test_size,
            "validation_size_within_development": validation_size_within_development,
            "random_state": random_state,
        },
        "environment": {
            "python_version": python_version(),
            "pandas_version": version("pandas"),
            "numpy_version": version("numpy"),
            "scipy_version": version("scipy"),
            "scikit-learn_version": version("scikit-learn"),
        },
    }

    output_path = Path("data/processed/split_assignments.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        assignments.to_csv(output_path, index=False, mode="x")
        print(f"Split assignments saved to {output_path}")
    else:
        existing = pd.read_csv(
            output_path,
            dtype=str,
            keep_default_na=False,
        )
        validate_split_assignments(existing, cohort)
        existing_sorted = existing.sort_values("encounter_id").reset_index(drop=True)
        assignments_sorted = assignments.sort_values("encounter_id").reset_index(drop=True)

        if not existing_sorted.equals(assignments_sorted):
            raise ValueError("Existing assignments differ from newly computed ones.")

        print(f"Existing assignments at {output_path} match; reusing them.")

    metadata_path = Path("data/processed/split_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        print(f"Metadata saved to {metadata_path}")
    else:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata_from_file = json.load(file)

        if metadata != metadata_from_file:
            raise ValueError("Existing metadata differ from newly computed ones.")

        print(f"Existing metadata at {metadata_path} match; reusing them.")

    3

if __name__ == "__main__":
    main()
