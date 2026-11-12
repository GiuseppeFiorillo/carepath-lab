import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def validate_split_assignments(
    assignments: pd.DataFrame, cohort: pd.DataFrame
) -> None:
    """Check partition labels, encounter/patient pairs and patient grouping."""
    partition_names = set(assignments["partition"])
    unexpected_partitions = partition_names - {"train", "validation", "test"}

    if unexpected_partitions:
        raise ValueError(
            f"Unexpected partition categories: {sorted(unexpected_partitions)}"
        )

    if not assignments["encounter_id"].is_unique:
        raise ValueError("Duplicate encounter IDs found in assignments.")

    partitions_per_patient = (
        assignments.groupby("patient_nbr")["partition"].nunique()
    )

    if (partitions_per_patient > 1).any():
        raise ValueError("Each patient must belong to exactly one partition.")

    if set(assignments["encounter_id"]) != set(cohort["encounter_id"]):
        raise ValueError("Assignment encounter IDs do not match the cohort.")

    assignment_pairs = set(
        zip(assignments["encounter_id"], assignments["patient_nbr"])
    )
    cohort_pairs = set(zip(cohort["encounter_id"], cohort["patient_nbr"]))

    if assignment_pairs != cohort_pairs:
        raise ValueError("Assignment encounter/patient pairs do not match the cohort.")


def split_patient_holdout(data, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )

    development_positions, test_positions = next(
        splitter.split(data, groups=data["patient_nbr"])
    )

    development = data.iloc[development_positions].copy()
    test = data.iloc[test_positions].copy()

    return development, test


def make_patient_folds(data, n_splits=5, random_state=42):
    """Yield positional CV indices, keeping patients together within each fold.

    Pass only the saved training partition, not the full analysis cohort.
    """
    splitter = GroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return splitter.split(data, groups=data["patient_nbr"])
