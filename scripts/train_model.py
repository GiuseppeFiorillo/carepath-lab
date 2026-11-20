"""Train the current 41-feature CatBoost and export checked validation predictions."""

import argparse
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from carepath_lab.data import file_sha256, load_verified_partitions
from carepath_lab.explainability import compute_native_shap
from carepath_lab.features import FEATURE_SETS, build_model_inputs
from carepath_lab.modeling import build_catboost_model, fit_and_predict_validation


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = Path("docs/reports/feature_review_selection.json")
    cv_path = Path("docs/reports/feature_review_folds.csv")
    selection = json.loads(selection_path.read_text())
    if selection["features"] != FEATURE_SETS["reviewed"]:
        raise ValueError("Reviewed feature configuration does not match the recorded selection.")
    if file_sha256(cv_path) != selection["fold_results_sha256"]:
        raise ValueError("Recorded CV evidence has changed since feature selection.")
    paths = [
        selection_path, cv_path,
        Path("data/raw/diabetic_data.csv"),
        Path("data/processed/split_assignments.csv"),
        Path("data/processed/split_metadata.json"),
        Path(__file__),
        *sorted(Path("carepath_lab").glob("*.py")),
    ]
    before = {str(path): file_sha256(path) for path in paths}
    train, validation = load_verified_partitions()
    config = FEATURE_SETS["reviewed"]
    imputed = ("medical_specialty",) if "medical_specialty" in config["categorical"] else ()
    builder = partial(build_catboost_model, imputed_categorical_features=imputed)
    pipeline, outcomes, probabilities = fit_and_predict_validation(
        train, validation, builder, feature_set="reviewed"
    )
    predictions = validation[["encounter_id", "patient_nbr"]].copy()
    predictions["outcome"] = outcomes
    predictions["catboost_specialty_mode"] = probabilities
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False, mode="x")
    model_path = output_dir / "selected_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    inputs, _ = build_model_inputs(validation, feature_set="reviewed")
    np.testing.assert_allclose(reloaded.predict_proba(inputs)[:, 1], probabilities, rtol=1e-12, atol=1e-12)
    classifier = pipeline.named_steps["classifier"]
    examples, contributions = compute_native_shap(
        pipeline, inputs.iloc[:5]
    )
    raw_scores = classifier.predict(
        examples,
        prediction_type="RawFormulaVal",
    )
    reconstructed = contributions.sum(axis=1)
    if before != {str(path): file_sha256(path) for path in paths}:
        raise ValueError("Inputs or code changed during training.")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": "reviewed", "features": config,
        "model": "CatBoost", "prediction_column": "catboost_specialty_mode",
        "imputed_categorical_features": list(imputed),
        "model_parameters": classifier.get_params(),
        "training_encounters": len(train), "validation_encounters": len(validation),
        "validation_patients": int(validation.patient_nbr.nunique()),
        "validation_positives": int(outcomes.sum()),
        "validation_prevalence": float(outcomes.mean()),
        "validation_metrics": {
            "ap": average_precision_score(outcomes, probabilities),
            "auroc": roc_auc_score(outcomes, probabilities),
            "brier": brier_score_loss(outcomes, probabilities),
        },
        "validation_status": "Descriptive development results after training-CV feature selection; no selection using these metrics",
        "final_test_scored": False,
        "shap_smoke_check": {
            "encounters": 5, "output": "raw log-odds",
            "maximum_additivity_error": float(np.max(np.abs(reconstructed - raw_scores))),
        },
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost", "joblib"]}},
        "input_sha256": before,
        "output_sha256": {path.name: file_sha256(path) for path in sorted(output_dir.iterdir())},
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(metadata["validation_metrics"], indent=2), flush=True)
    print(f"Saved current pipeline and validation predictions to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
