"""Audit saved validation predictions without refitting the selected model."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from carepath_lab.audit import (
    bootstrap_gender_recall_difference,
    build_audit_table,
    summarize_subgroup_calibration,
    summarize_subgroup_metrics,
)
from scripts.train_baseline import load_verified_partitions
from scripts.tune_lightgbm import file_sha256


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()

    input_paths = [
        Path("data/raw/diabetic_data.csv"),
        Path("data/processed/split_assignments.csv"),
        Path("data/processed/split_metadata.json"),
        Path("artifacts/phase2_fixed_validation_2026-11-12/validation_predictions.csv"),
        Path("artifacts/phase2_fixed_validation_2026-11-12/metadata.json"),
        Path("scripts/train_baseline.py"),
        Path("scripts/audit_model.py"),
        *sorted(Path("carepath_lab").glob("*.py")),
    ]
    input_hashes = {str(path): file_sha256(path) for path in input_paths}

    _, validation = load_verified_partitions()

    scores = pd.read_csv(
        Path("artifacts/phase2_fixed_validation_2026-11-12")
        / "validation_predictions.csv",
        dtype={"encounter_id": str, "patient_nbr": str},
    )

    audit = build_audit_table(validation, scores)
    metrics = summarize_subgroup_metrics(audit)
    calibration = summarize_subgroup_calibration(audit)

    print(metrics.to_string(index=False))
    print("\nSubgroup calibration")
    print(calibration.to_string(index=False))

    metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    calibration.to_csv(
        output_dir / "subgroup_calibration.csv",
        index=False,
    )

    bootstrap_config = {
        "n_bootstrap": 1000,
        "random_state": 123,
    }

    differences = bootstrap_gender_recall_difference(
        audit, **bootstrap_config
    )
    valid_count = np.isfinite(differences).sum()

    print(
        f"\nBootstrap recall difference: "
        f"{valid_count}/{len(differences)} finite replicates"
    )

    if valid_count == len(differences):
        lower, upper = np.quantile(differences, [0.025, 0.975])
        print(
            "Recall Female - Male | threshold=0.20\n"
            f"95% patient-cluster bootstrap percentile CI: "
            f"[{100 * lower:.2f}, {100 * upper:.2f}] percentage points"
        )

        observed_recalls = metrics.loc[
            metrics["group_column"].eq("gender")
            & metrics["threshold"].eq(0.20)
        ].set_index("group_value")["recall"]

        bootstrap_summary = pd.DataFrame([{
            "comparison": "Female - Male",
            "metric": "recall",
            "threshold": 0.20,
            "observed_difference": (
                observed_recalls["Female"] - observed_recalls["Male"]
            ),
            "ci_lower": lower,
            "ci_upper": upper,
            "replicates": len(differences),
            "finite_replicates": int(valid_count),
        }])

        bootstrap_summary.to_csv(
            output_dir / "bootstrap_recall_summary.csv",
            index=False,
        )
    else:
        print("CI not reported: some bootstrap differences are undefined.")

    metadata = {
        "started_at_utc": started_at,
        "evaluation_partition": "validation",
        "model": "catboost_specialty_mode",
        "source_artifacts": "artifacts/phase2_fixed_validation_2026-11-12",
        "group_columns": ["gender", "age", "race"],
        "thresholds": [0.10, 0.20, 0.30, 0.50],
        "calibration_bin_edges": [0, 0.10, 0.20, 0.30, 0.40, 0.50, 1],
        "validation_status": "Repeatedly inspected development validation",
        "bootstrap": {
            **bootstrap_config,
            "method": "patient-cluster percentile",
            "comparison": "Female - Male",
            "metric": "recall",
            "threshold": 0.20,
            "confidence_level": 0.95,
            "finite_replicates": int(valid_count),
            "ci_reported": bool(valid_count == len(differences)),
            "conditional_on_fitted_model": True,
            "simultaneous_coverage": False,
            "units": "proportion",
        },
    }

    if input_hashes != {
        str(path): file_sha256(path) for path in input_paths
    }:
        raise ValueError("Inputs or code changed during the audit.")

    metadata["input_sha256"] = input_hashes
    metadata["output_sha256"] = {
        path.name: file_sha256(path)
        for path in sorted(output_dir.glob("*.csv"))
    }
    metadata["environment"] = {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn"]
        },
    }
    metadata["final_test_scored"] = False

    with (output_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
