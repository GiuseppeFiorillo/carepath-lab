"""Resample patient groups while keeping their encounter rows together."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.random import Generator
from sklearn.metrics import brier_score_loss


def group_positions_by_patient(patient_ids: Iterable[str]) -> dict[str, list[int]]:
    """Map each patient to zero-based encounter positions in the current order."""
    positions_by_patient = {}
    for position, patient_nbr in enumerate(patient_ids):
        if patient_nbr not in positions_by_patient:
            positions_by_patient[patient_nbr] = []
        positions_by_patient[patient_nbr].append(position)
    return positions_by_patient


def sample_patient_positions(
    positions_by_patient: dict[str, list[int]], rng: Generator
) -> list[int]:
    """Draw patient blocks with replacement, preserving every draw's positions."""
    patients = list(positions_by_patient)
    sampled_patients = rng.choice(
        patients,
        size=len(patients),
        replace=True,
    )
    sampled_positions = []

    for patient in sampled_patients:
        sampled_positions.extend(positions_by_patient[patient])

    return sampled_positions


def bootstrap_metric_scores(
    outcomes: pd.Series,
    predictions: dict[str, np.ndarray],
    patient_numbers: pd.Series,
    n_bootstrap: int = 1000,
    random_state: int = 123,
    score_function=None,
) -> pd.DataFrame:
    """Return per-model metric scores from shared patient-cluster bootstrap draws.

    Outcomes and patient numbers must have identical indices in identical order.
    Each prediction vector must follow that encounter order. Models stay fixed;
    only validation patient blocks are resampled, including repeated draws.
    The scorer receives outcomes and positive probabilities and must return a
    finite scalar. Omission selects Brier loss. Invalid scores raise rather than
    silently reducing the number of replicates used for interval estimation.
    """
    if not outcomes.index.equals(patient_numbers.index):
        raise ValueError("Outcome and patient indices must match in the same order.")
    if outcomes.empty or not predictions:
        raise ValueError("At least one encounter and one model are required.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    for name, probabilities in predictions.items():
        if probabilities.ndim != 1 or len(probabilities) != len(outcomes):
            raise ValueError(f"Prediction vector for {name} must match the outcome rows.")
    if score_function is None:
        score_function = brier_score_loss

    positions_by_patient = group_positions_by_patient(patient_numbers)
    rng = np.random.default_rng(random_state)
    bootstrap_rows = []

    for _ in range(n_bootstrap):
        positions = sample_patient_positions(positions_by_patient, rng)
        sampled_outcomes = outcomes.iloc[positions]
        scores = {}

        for name, probabilities in predictions.items():
            score = score_function(
                sampled_outcomes, probabilities[positions]
            )
            if not np.isscalar(score) or not np.isfinite(score):
                raise ValueError(f"Score for {name} must be a finite scalar.")
            scores[name] = float(score)

        bootstrap_rows.append(scores)

    return pd.DataFrame(bootstrap_rows)
