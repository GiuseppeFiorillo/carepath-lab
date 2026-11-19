import numpy as np
import pandas as pd

from carepath_lab.uncertainty import (
    group_positions_by_patient,
    sample_patient_positions,
)


def build_audit_table(
    validation: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Align saved predictions with validation demographics and outcomes."""

    demographics = validation[
        ["encounter_id", "patient_nbr", "age", "gender", "race", "readmitted"]
    ].copy()

    audit = demographics.merge(
        scores,
        on="encounter_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_source", "_saved"),
        indicator=True,
    )

    if not audit["_merge"].eq("both").all():
        raise ValueError("Validation and prediction encounter IDs must match exactly.")

    if not audit["patient_nbr_source"].eq(audit["patient_nbr_saved"]).all():
        raise ValueError("Patient IDs disagree between validation and predictions.")

    expected_outcomes = audit["readmitted"].map({
        "<30": 1,
        ">30": 0,
        "NO": 0,
    })

    if not expected_outcomes.eq(audit["outcome"]).all():
        raise ValueError("Recorded outcomes disagree between validation and predictions.")

    return audit


def summarize_subgroup_metrics(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize subgroup support and errors at fixed exploratory thresholds."""

    probabilities = audit["catboost_specialty_mode"]

    if not probabilities.between(0, 1).all():
        raise ValueError("Predicted probabilities must be non-missing and within [0, 1].")

    if not audit["outcome"].isin([0, 1]).all():
        raise ValueError("Outcomes must be non-missing binary values: 0 or 1.")

    rows = []

    for group_column in ["gender", "age", "race"]:
        for threshold in [0.10, 0.20, 0.30, 0.50]:
            for group_value, group in audit.groupby(
                group_column, dropna=False
            ):
                predicted_positive = (
                    group["catboost_specialty_mode"] >= threshold
                )
                tp = (predicted_positive & group["outcome"].eq(1)).sum()
                fn = (~predicted_positive & group["outcome"].eq(1)).sum()
                fp = (predicted_positive & group["outcome"].eq(0)).sum()
                tn = (~predicted_positive & group["outcome"].eq(0)).sum()
                assert tp + fn + fp + tn == len(group)
                encounter_count = len(group)
                patient_count = group["patient_nbr_source"].nunique()
                positive_count = tp + fn
                recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
                precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
                fpr = fp / (fp + tn) if fp + tn > 0 else float("nan")
                rows.append({
                    "threshold": threshold,
                    "group_column": group_column,
                    "group_value": group_value,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "recall": recall,
                    "precision": precision,
                    "fpr": fpr,
                    "encounter_count": encounter_count,
                    "patient_count": patient_count,
                    "positive_count": positive_count,
                })

    return pd.DataFrame(rows)


def summarize_subgroup_calibration(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibration within fixed probability bins and subgroups."""

    probabilities = audit["catboost_specialty_mode"]

    if not probabilities.between(0, 1).all():
        raise ValueError("Predicted probabilities must be non-missing and within [0, 1].")

    if not audit["outcome"].isin([0, 1]).all():
        raise ValueError("Outcomes must be non-missing binary values: 0 or 1.")

    calibration_data = audit.copy()
    calibration_data["probability_bin"] = pd.cut(
        audit["catboost_specialty_mode"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 1],
        include_lowest=True,
    )

    calibration_tables = []

    for group_column in ["gender", "age", "race"]:
        summary = calibration_data.groupby(
            [group_column, "probability_bin"],
            observed=True,
            dropna=False,
        ).agg(
            encounter_count=("outcome", "size"),
            patient_count=("patient_nbr_source", "nunique"),
            positive_count=("outcome", "sum"),
            mean_predicted_probability=("catboost_specialty_mode", "mean"),
            observed_positive_fraction=("outcome", "mean"),
        )

        summary["prediction_minus_observed"] = (
            summary["mean_predicted_probability"]
            - summary["observed_positive_fraction"]
        )

        summary = summary.reset_index().rename(
            columns={group_column: "group_value"}
        )

        summary["group_column"] = group_column
        calibration_tables.append(summary)

    return pd.concat(
        calibration_tables, ignore_index=True
    )


def bootstrap_gender_recall_difference(
    audit: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 123,
) -> np.ndarray:
    """Resample patients and return Female-minus-Male recall differences at 0.20."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    patient_positions = group_positions_by_patient(audit["patient_nbr_source"])
    recall_differences = []
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstrap):
        positions = sample_patient_positions(patient_positions, rng)
        replica = audit.iloc[positions]
        metrics = summarize_subgroup_metrics(replica)

        gender_metrics = metrics.loc[
            metrics["group_column"].eq("gender")
            & metrics["threshold"].eq(0.20)
        ].set_index("group_value")

        recalls = gender_metrics["recall"].reindex(["Female", "Male"])
        recall_differences.append(recalls["Female"] - recalls["Male"])

    return np.asarray(recall_differences)
