"""Export the frozen Phase 2 comparison on inspected development validation.

This command never scores the final test. Patient bootstrap intervals condition
on fitted models and do not include training or model-selection uncertainty.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay, auc, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_auc_score,
)

from carepath_lab.features import FEATURE_SETS
from carepath_lab.modeling import (
    build_baseline_model, build_catboost_model, build_lightgbm_model,
    build_random_forest_model, fit_and_predict_validation,
)
from carepath_lab.uncertainty import bootstrap_metric_scores
from scripts.train_baseline import load_verified_partitions
from scripts.tune_lightgbm import file_sha256


def final_builders():
    """Return procedures fixed by training CV before this evaluation."""
    return {
        "logistic": build_baseline_model,
        "random_forest": build_random_forest_model,
        "lightgbm_selected": partial(
            build_lightgbm_model, num_leaves=15, min_child_samples=100, reg_lambda=10.0,
        ),
        "catboost_specialty_mode": partial(
            build_catboost_model, imputed_categorical_features=("medical_specialty",),
            add_missing_indicators=False,
        ),
    }


def calibration_table(outcomes, probabilities, patients):
    """Summarize fixed bins without fitting a calibration mapping."""
    data = pd.DataFrame({
        "probability": probabilities, "outcome": outcomes.to_numpy(),
        "patient": patients.to_numpy(),
    })
    data["probability_bin"] = pd.cut(
        data["probability"], [0, .1, .2, .3, .4, .5, 1], include_lowest=True,
    )
    return data.groupby("probability_bin", observed=True).agg(
        encounter_count=("outcome", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_positive_fraction=("outcome", "mean"),
        observed_positive_count=("outcome", "sum"),
        patient_count=("patient", "nunique"),
    )


def main(output_dir: Path, n_bootstrap: int = 1000) -> None:
    """Fit only saved training and export a fixed development comparison."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [Path(name) for name in [
        "data/raw/diabetic_data.csv", "data/processed/split_assignments.csv",
        "data/processed/split_metadata.json", "carepath_lab/features.py",
        "carepath_lab/modeling.py", "carepath_lab/uncertainty.py",
        "carepath_lab/splitting.py", "scripts/train_baseline.py",
        "scripts/evaluate_final_models.py",
    ]]
    before = {str(path): file_sha256(path) for path in inputs}
    train, validation = load_verified_partitions()
    started_at = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    predictions, calibrations, model_parameters = {}, {}, {}
    rows = []
    for name, builder in final_builders().items():
        print(f"Fitting frozen procedure: {name}", flush=True)
        model, outcomes, probabilities = fit_and_predict_validation(
            train, validation, builder, feature_set="extended",
        )
        predictions[name] = probabilities
        model_parameters[name] = model.named_steps["classifier"].get_params()
        calibrations[name] = calibration_table(outcomes, probabilities, validation.patient_nbr)
        calibrations[name].to_csv(output_dir / f"calibration_{name}.csv", mode="x")
        if name == "catboost_specialty_mode":
            joblib.dump(model, output_dir / "selected_pipeline.joblib")

    dummy = DummyClassifier(strategy="prior").fit(
        np.zeros((len(train), 1)), train.readmitted.eq("<30").astype(int),
    )
    predictions["dummy"] = dummy.predict_proba(np.zeros((len(validation), 1)))[:, 1]
    for name, probabilities in predictions.items():
        precision, recall, _ = precision_recall_curve(outcomes, probabilities)
        rows.append({
            "model": name, "encounters": len(outcomes),
            "positives": int(outcomes.sum()), "prevalence": float(outcomes.mean()),
            "ap": average_precision_score(outcomes, probabilities),
            "auroc": roc_auc_score(outcomes, probabilities),
            "brier": brier_score_loss(outcomes, probabilities),
            "pr_auc_trapezoidal": auc(recall, precision),
        })
    metrics = pd.DataFrame(rows).set_index("model")
    metrics.to_csv(output_dir / "metrics.csv", mode="x")
    pd.DataFrame({
        "encounter_id": validation.encounter_id.to_numpy(),
        "patient_nbr": validation.patient_nbr.to_numpy(),
        "outcome": outcomes.to_numpy(), **predictions,
    }).to_csv(output_dir / "validation_predictions.csv", index=False, mode="x")

    intervals, paired = [], []
    for metric, scorer in [
        ("brier", brier_score_loss), ("ap", average_precision_score),
        ("auroc", roc_auc_score),
    ]:
        print(f"Patient-cluster bootstrap: {metric}, {n_bootstrap} shared draws", flush=True)
        scores = bootstrap_metric_scores(
            outcomes, predictions, validation.patient_nbr,
            n_bootstrap=n_bootstrap, random_state=123, score_function=scorer,
        )
        scores.to_csv(output_dir / f"bootstrap_{metric}.csv", index=False, mode="x")
        for name in predictions:
            low, high = scores[name].quantile([.025, .975])
            intervals.append({"model": name, "metric": metric,
                              "point": metrics.loc[name, metric], "lower": low, "upper": high})
        for first, second in combinations(predictions, 2):
            low, high = (scores[first] - scores[second]).quantile([.025, .975])
            paired.append({"first": first, "second": second, "metric": metric,
                           "point": metrics.loc[first, metric] - metrics.loc[second, metric],
                           "lower": low, "upper": high})
    pd.DataFrame(intervals).to_csv(output_dir / "intervals.csv", index=False, mode="x")
    pd.DataFrame(paired).to_csv(output_dir / "paired_intervals.csv", index=False, mode="x")

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="carepath-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for name, probabilities in predictions.items():
        if name != "dummy":
            PrecisionRecallDisplay.from_predictions(
                outcomes, probabilities, name=name, ax=ax,
            )
    ax.axhline(outcomes.mean(), color="gray", linestyle="--", label="Validation prevalence")
    ax.set_title("Phase 2: inspected development validation")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "precision_recall.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for (name, table), ax in zip(calibrations.items(), axes.flat):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(table.mean_predicted_probability, table.observed_positive_fraction, "o-")
        for row in table.itertuples():
            ax.annotate(str(row.encounter_count),
                        (row.mean_predicted_probability, row.observed_positive_fraction),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name,
               xlabel="Mean predicted probability", ylabel="Observed fraction")
    fig.suptitle("Descriptive calibration; labels are encounter counts, not independent patients")
    fig.savefig(output_dir / "calibration.png", dpi=180)
    plt.close(fig)

    if before != {str(path): file_sha256(path) for path in inputs}:
        raise ValueError("Inputs or code changed during evaluation.")
    metadata = {
        "started_at_utc": started_at, "elapsed_seconds": perf_counter() - start,
        "feature_set": "extended", "features": FEATURE_SETS["extended"],
        "model_parameters": model_parameters,
        "preprocessing": {
            "logistic": "standard scaling and one-hot",
            "random_forest": "raw numerics and one-hot",
            "lightgbm_selected": "raw numerics and one-hot",
            "catboost_specialty_mode": "native categories, training-mode specialty imputation, no indicator/scaling",
        },
        "selected_before_evaluation": "catboost_specialty_mode",
        "training_encounters": len(train), "validation_encounters": len(validation),
        "validation_patients": int(validation.patient_nbr.nunique()),
        "validation_status": "Repeatedly inspected development data; not confirmatory external validation",
        "final_test_scored": False,
        "bootstrap": {"replicates": n_bootstrap, "seed": 123,
                      "unit": "patient", "method": "paired percentile",
                      "conditional_on_fitted_models": True, "simultaneous_coverage": False},
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in [
                "pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "catboost", "matplotlib", "joblib",
            ]}},
        "file_sha256": {**before, **{
            str(path): file_sha256(path) for path in output_dir.iterdir() if path.is_file()
        }},
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print(f"Saved frozen comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    main(args.output_dir, n_bootstrap=args.bootstrap)
