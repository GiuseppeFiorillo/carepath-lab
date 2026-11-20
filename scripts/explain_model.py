"""Export reproducible native SHAP explanations for current-model validation encounters."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import matplotlib.pyplot as plt
import numpy as np

from carepath_lab.data import file_sha256, load_verified_partitions
from carepath_lab.explainability import (
    compute_native_shap,
    plot_beeswarm,
    plot_global_importance,
    plot_waterfall,
)
from carepath_lab.features import build_model_inputs


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = Path("artifacts/current_model")
    model_path = model_dir / "selected_pipeline.joblib"
    source_paths = [
        Path("data/raw/diabetic_data.csv"),
        Path("data/processed/split_assignments.csv"),
        Path("data/processed/split_metadata.json"),
    ]
    input_paths = [
        *source_paths, model_path, model_dir / "metadata.json", Path(__file__),
        *sorted(Path("carepath_lab").glob("*.py")),
    ]
    input_hashes = {str(path): file_sha256(path) for path in input_paths}
    model_metadata = json.loads((model_dir / "metadata.json").read_text())
    if model_metadata["feature_set"] != "reviewed":
        raise ValueError("The current model must use the reviewed feature set.")
    if input_hashes[str(model_path)] != model_metadata["output_sha256"][model_path.name]:
        raise ValueError("The model fingerprint differs from its training metadata.")
    for path in source_paths:
        if input_hashes[str(path)] != model_metadata["input_sha256"][str(path)]:
            raise ValueError("Source or split fingerprints differ from model training.")
    pipeline = joblib.load(model_path)

    _, validation = load_verified_partitions()
    X_validation, _ = build_model_inputs(validation, "reviewed")

    sampling_config = {"n": 1000, "random_state": 42}
    examples = X_validation.sample(**sampling_config).copy()
    transformed_inputs, shap_values = compute_native_shap(pipeline, examples)

    print(transformed_inputs.shape)
    print(shap_values.shape)

    sample_info = validation.loc[
        examples.index,
        ["encounter_id", "patient_nbr"],
    ].copy()

    assert sample_info.index.equals(transformed_inputs.index)
    assert sample_info["encounter_id"].is_unique

    print("Sample identifiers:", sample_info.shape)

    sample_info.to_csv(output_dir / "sample.csv", index=False)
    transformed_inputs.to_csv(
        output_dir / "transformed_inputs.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir / "shap_values.npz",
        contributions=shap_values[:, :-1],
        reference_values=shap_values[:, -1],
        feature_names=np.asarray(transformed_inputs.columns, dtype=str),
    )

    figure = plot_global_importance(
        shap_values[:, :-1],
        transformed_inputs.columns.tolist(),
    )
    figure.savefig(output_dir / "global_importance.png", dpi=180)
    plt.close(figure)

    figure = plot_beeswarm(
        contributions=shap_values[:, :-1],
        reference_values=shap_values[:, -1],
        feature_names=transformed_inputs.columns.tolist(),
    )
    figure.savefig(output_dir / "beeswarm.png", dpi=180)
    plt.close(figure)

    local_position = 0

    figure = plot_waterfall(
        contributions=shap_values[local_position, :-1],
        reference_value=shap_values[local_position, -1],
        feature_values=transformed_inputs.iloc[local_position].to_numpy(),
        feature_names=transformed_inputs.columns.tolist(),
    )
    figure.savefig(output_dir / "waterfall.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    if input_hashes != {str(path): file_sha256(path) for path in input_paths}:
        raise ValueError("Inputs or code changed during explanation.")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "feature_set": "reviewed",
        "feature_names": transformed_inputs.columns.tolist(),
        "sampling": {**sampling_config, "unit": "encounter"},
        "evaluation_partition": "repeatedly inspected development validation",
        "row_order": "sample.csv rows align with transformed_inputs.csv and both SHAP arrays",
        "shap": {
            "implementation": "CatBoost native ShapValues, default calculation",
            "output_units": "raw_log_odds", "background_data_provided": False,
            "reference": "native model reference, not sample mean probability",
            "numerical_checks_passed": True, "rtol": 1e-6, "atol": 1e-8,
        },
        "waterfall": {
            "sample_row": local_position,
            "selection": "first row of the fixed random validation sample",
        },
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib", "matplotlib", "shap"]
        }},
        "input_sha256": input_hashes,
        "output_sha256": {path.name: file_sha256(path) for path in sorted(output_dir.iterdir())},
        "limitations": [
            "Encounter-weighted sample, not patient-balanced or population representative.",
            "Predictive attribution is not causal; correlated features affect attribution.",
            "Numerical reconstruction does not establish calibration or clinical validity.",
        ],
        "final_test_scored": False,
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Saved checked SHAP explanations to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
