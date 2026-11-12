"""Verify saved split provenance and fit the initial baseline on training only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from carepath_lab.cohort import select_analysis_cohort
from carepath_lab.features import attach_split_assignments, build_model_inputs
from carepath_lab.modeling import build_baseline_model
from carepath_lab.splitting import validate_split_assignments
from carepath_lab.uncertainty import group_positions_by_patient, sample_patient_positions


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

    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

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


def print_section(title: str) -> None:
    """Separate console output without changing individual result labels."""
    print(f"\n{title}\n{'-' * len(title)}")


def print_calibration_summary(
    outcomes: pd.Series,
    positive_probabilities: np.ndarray,
    patient_numbers: pd.Series,
) -> None:
    """Describe fixed predictions by bin, retaining encounter and patient support.

    Series must share their encounter index; probability order must match it.
    """
    calibration_data = pd.DataFrame(
        {
            "predicted_probability": positive_probabilities,
            "observed_outcome": outcomes,
            "patient_nbr": patient_numbers,
        },
        index=outcomes.index,
    )
    calibration_data["probability_bin"] = pd.cut(
        calibration_data["predicted_probability"],
        bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        include_lowest=True,
    )
    calibration_summary = calibration_data.groupby(
        "probability_bin", observed=True
    ).agg(
        encounter_count=("observed_outcome", "size"),
        mean_predicted_probability=("predicted_probability", "mean"),
        observed_positive_fraction=("observed_outcome", "mean"),
        observed_positive_count=("observed_outcome", "sum"),
        patient_count=("patient_nbr", "nunique"),
    )
    print("Validation calibration summary (fixed probability bins):")
    print(calibration_summary.to_string(float_format=lambda value: f"{value:.6f}"))


def main(n_bootstrap: int = 1000) -> None:
    """Fit training-only references and inspect validation performance."""
    # 1. Verify provenance and construct model inputs from saved partitions.
    train, validation = load_verified_partitions()
    X_train, y_train = build_model_inputs(train)
    X_validation, y_validation = build_model_inputs(validation)

    print_section("1. Model inputs")
    print(f"Training: X {X_train.shape}, y {y_train.shape}")
    print(f"Validation: X {X_validation.shape}, y {y_validation.shape}")

    # 2. Fit both preprocessing and classifier using only training encounters.
    print_section("2. Training and validation predictions")
    model = build_baseline_model()
    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]

    print("Learned classes:", classifier.classes_)
    print("Optimizer iterations:", classifier.n_iter_)

    # Predict on validation without refitting; columns follow classifier.classes_.
    validation_probabilities = model.predict_proba(X_validation)
    print("Validation probabilities shape:", validation_probabilities.shape)

    validation_positive_probabilities = validation_probabilities[:, 1]

    print(
        "Validation positive probabilities shape:",
        validation_positive_probabilities.shape,
    )
    print(
        "Validation positive probabilities: "
        f"min={validation_positive_probabilities.min():.6f}, "
        f"max={validation_positive_probabilities.max():.6f}"
    )

    # Compare with a feature-independent reference fitted on training labels.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_probabilities = dummy.predict_proba(X_validation)
    dummy_positive_probabilities = dummy_probabilities[:, 1]
    print("Dummy positive probabilities shape:", dummy_positive_probabilities.shape)
    print(
        "Dummy positive probabilities: "
        f"min={dummy_positive_probabilities.min():.6f}, "
        f"max={dummy_positive_probabilities.max():.6f}"
    )

    # 3. Compare probability loss and ranking on the same validation encounters.
    print_section("3. Validation probability and ranking metrics")
    logistic_brier_score = brier_score_loss(
        y_validation, validation_positive_probabilities
    )
    dummy_brier_score = brier_score_loss(y_validation, dummy_positive_probabilities)

    print(f"Validation Brier score (logistic): {logistic_brier_score:.6f}")
    print(f"Validation Brier score (dummy): {dummy_brier_score:.6f}")

    # Evaluate ranking with continuous scores, using both outcome classes.
    logistic_auroc_score = roc_auc_score(
        y_validation, validation_positive_probabilities
    )
    dummy_auroc_score = roc_auc_score(y_validation, dummy_positive_probabilities)

    print(f"Validation AUROC score (logistic): {logistic_auroc_score:.6f}")
    print(f"Validation AUROC score (dummy): {dummy_auroc_score:.6f}")

    # Use non-interpolated average precision, distinct from trapezoidal PR area.
    logistic_ap_score = average_precision_score(
        y_validation, validation_positive_probabilities
    )
    dummy_ap_score = average_precision_score(y_validation, dummy_positive_probabilities)

    print(f"Validation AP score (logistic): {logistic_ap_score:.6f}")
    print(f"Validation AP score (dummy): {dummy_ap_score:.6f}")

    # 4. Inspect PR coordinates and explicitly label the area convention.
    print_section("4. Precision–recall curve")
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_validation, validation_positive_probabilities
    )

    print("Validation precision–recall curve summary:")
    print(
        f"Shapes: precision={pr_precision.shape}, recall={pr_recall.shape}, "
        f"thresholds={pr_thresholds.shape}"
    )
    print(
        "First pair (precision, recall): "
        f"({pr_precision[0]:.6f}, {pr_recall[0]:.6f})"
    )
    print(
        "Last pair (precision, recall): "
        f"({pr_precision[-1]:.6f}, {pr_recall[-1]:.6f})"
    )

    logistic_pr_auc = auc(pr_recall, pr_precision)
    print(f"Validation PR-AUC (logistic, trapezoidal): {logistic_pr_auc:.6f}")

    # 5. Compare illustrative thresholds; these are not clinical recommendations.
    print_section("5. Illustrative decision thresholds")
    for threshold in [0.5, 0.2]:
        validation_predictions = (
            validation_positive_probabilities >= threshold
        ).astype(int)
        validation_confusion_matrix = confusion_matrix(
            y_validation,
            validation_predictions,
            labels=[0, 1],
        )

        print(f"Validation confusion matrix (logistic, threshold={threshold:.2f}):")
        print(validation_confusion_matrix)

        _, fp, fn, tp = validation_confusion_matrix.ravel()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print(f"Validation precision (threshold={threshold:.2f}): {precision:.2%}")
        print(f"Validation recall (threshold={threshold:.2f}): {recall:.2%}")

    # 6. Describe calibration without fitting or changing predicted probabilities.
    print_section("6. Calibration by probability bin")
    print_calibration_summary(
        y_validation, validation_positive_probabilities, validation["patient_nbr"]
    )

    # 7. Resample validation patient blocks with the fitted models held fixed.
    print_section("7. Patient-cluster bootstrap uncertainty")
    positions_by_patient = group_positions_by_patient(validation["patient_nbr"])
    rng = np.random.default_rng(123)

    bootstrap_brier_scores = []
    bootstrap_brier_differences = []

    for _ in range(n_bootstrap):
        sampled_positions = sample_patient_positions(positions_by_patient, rng)

        sampled_outcome = y_validation.iloc[sampled_positions]
        sampled_positive_probabilities = validation_positive_probabilities[sampled_positions]

        bootstrap_brier_score = brier_score_loss(
            sampled_outcome, sampled_positive_probabilities
        )
        bootstrap_brier_scores.append(bootstrap_brier_score)

        sampled_dummy_positive_probabilities = dummy_positive_probabilities[sampled_positions]
        bootstrap_dummy_brier_score = brier_score_loss(
            sampled_outcome, sampled_dummy_positive_probabilities
        )
        bootstrap_brier_differences.append(
            bootstrap_brier_score - bootstrap_dummy_brier_score
        )

    print(f"Bootstrap Brier summary ({n_bootstrap} patient-cluster replicates):")
    print(
        f"Bootstrap replicate Brier mean: {np.mean(bootstrap_brier_scores):.6f}\n"
        f"Bootstrap replicate Brier min: {min(bootstrap_brier_scores):.6f}\n"
        f"Bootstrap replicate Brier max: {max(bootstrap_brier_scores):.6f}\n"
    )

    lower, upper = np.quantile(
        bootstrap_brier_scores,
        [0.025, 0.975],
    )

    print(
        "Validation Brier (logistic), 95% patient-cluster bootstrap percentile CI: "
        f"[{lower:.6f}, {upper:.6f}]"
    )

    lower_diff, upper_diff = np.quantile(
        bootstrap_brier_differences,
        [0.025, 0.975],
    )

    print(
        "Original logistic-dummy Brier difference: "
        f"{logistic_brier_score - dummy_brier_score:.6f}\n"
        "Validation Brier (logistic-dummy difference), 95% patient-cluster bootstrap percentile CI: "
        f"[{lower_diff:.6f}, {upper_diff:.6f}]"
    )


if __name__ == "__main__":
    main()
