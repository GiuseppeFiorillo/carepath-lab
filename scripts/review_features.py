"""Review declared feature exclusions on patient-grouped training folds."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from carepath_lab.features import FEATURE_SETS
from carepath_lab.modeling import build_catboost_model, evaluate_model_cv
from scripts.train_baseline import load_verified_partitions
from scripts.tune_lightgbm import file_sha256


CONSTANT_FEATURES = ["acetohexamide", "examide", "citoglipton"]

EXCLUSIONS = {
    "reference": [],
    "no_constants": CONSTANT_FEATURES,
    "no_payer": [*CONSTANT_FEATURES, "payer_code"],
    "no_specialty": [*CONSTANT_FEATURES, "medical_specialty"],
    "no_payer_specialty": [*CONSTANT_FEATURES, "payer_code", "medical_specialty"],
    "no_labs": [*CONSTANT_FEATURES, "max_glu_serum", "A1Cresult"],
    "no_diagnoses": [*CONSTANT_FEATURES, "diag_1", "diag_2", "diag_3"],
    "no_demographics": [*CONSTANT_FEATURES, "race", "gender"],
}


def make_review_profiles(train):
    for name in CONSTANT_FEATURES:
        if train[name].nunique(dropna=False) != 1:
            raise ValueError(f"Declared constant {name} varies in training.")
    return {
        profile: {
            kind: [name for name in names if name not in excluded]
            for kind, names in FEATURE_SETS["extended"].items()
        }
        for profile, excluded in EXCLUSIONS.items()
    }


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        Path("data/raw/diabetic_data.csv"),
        Path("data/processed/split_assignments.csv"),
        Path("data/processed/split_metadata.json"),
        Path(__file__), Path("scripts/train_baseline.py"),
        *sorted(Path("carepath_lab").glob("*.py")),
    ]
    hashes = {str(path): file_sha256(path) for path in paths}
    train, _ = load_verified_partitions()
    profiles = make_review_profiles(train)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_partition": "saved training only",
        "cv": {"method": "GroupKFold", "n_splits": 5, "shuffle": True, "random_state": 42, "group": "patient_nbr"},
        "model": {"iterations": 200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "random_seed": 42, "thread_count": 8},
        "decision_policy": {
            "reference": "no_constants", "maximum_mean_ap_loss": 0.002,
            "maximum_mean_brier_increase": 0.0002,
            "interpretation": "Exploratory engineering tolerances, not equivalence margins; verify joint removal once.",
        },
        "profiles": deepcopy(profiles),
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost"]}},
        "input_sha256": hashes,
        "final_test_scored": False,
    }
    profile_path = output_dir / "protocol.json"
    profile_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    rows = []
    # Runtime-only configurations avoid adding a permanent feature set per ablation.
    temporary_key = "_feature_review"
    if temporary_key in FEATURE_SETS:
        raise ValueError("Temporary review configuration already exists.")
    try:
        for name, config in profiles.items():
            FEATURE_SETS[temporary_key] = config
            imputed = ("medical_specialty",) if "medical_specialty" in config["categorical"] else ()
            builder = partial(build_catboost_model, imputed_categorical_features=imputed)
            print(f"Evaluating {name}: {sum(map(len, config.values()))} predictors", flush=True)
            result = evaluate_model_cv(train, feature_set=temporary_key, model_builder=builder)
            result.insert(0, "profile", name)
            result.to_csv(output_dir / f"{name}.csv", index=False, mode="x")
            rows.append(result)
            print(result[["ap", "auroc", "brier"]].mean().to_string(), flush=True)
    finally:
        FEATURE_SETS.pop(temporary_key, None)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "fold_results.csv", index=False, mode="x")
    summary = combined.groupby("profile").agg(
        mean_ap=("ap", "mean"), sd_ap=("ap", "std"),
        mean_auroc=("auroc", "mean"), mean_brier=("brier", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", mode="x")
    if hashes != {str(path): file_sha256(path) for path in paths}:
        raise ValueError("Inputs or code changed during the feature review.")
    metadata["output_sha256"] = {path.name: file_sha256(path) for path in sorted(output_dir.glob("*.csv"))}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
