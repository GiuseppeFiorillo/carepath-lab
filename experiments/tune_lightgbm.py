"""Run a bounded LightGBM search on patient-grouped training folds."""

import argparse
from functools import partial
from itertools import product
import platform

from copy import deepcopy
from importlib.metadata import version
import json
import pandas as pd
from pathlib import Path

from carepath_lab.features import FEATURE_SETS
from carepath_lab.modeling import build_lightgbm_model, evaluate_model_cv
from carepath_lab.data import file_sha256, load_verified_partitions


def run_search(
    train: pd.DataFrame,
    configurations: list[dict],
) -> pd.DataFrame:
    all_fold_results = []

    for configuration_id, configuration in enumerate(configurations, start=1):
        configured_builder = partial(
            build_lightgbm_model,
            **configuration,
        )

        fold_results = evaluate_model_cv(
            train,
            feature_set="extended",
            model_builder=configured_builder,
        )
        fold_results["configuration_id"] = configuration_id
        for name, value in configuration.items():
            fold_results[name] = value

        all_fold_results.append(fold_results)

    return pd.concat(all_fold_results, ignore_index=True)


def summarize_search(results: pd.DataFrame) -> pd.DataFrame:
    configuration_columns = [
        "configuration_id",
        "num_leaves",
        "min_child_samples",
        "reg_lambda",
    ]

    tuning_summary = (
        results
        .groupby(configuration_columns, as_index=False)
        .agg(
            mean_ap=("ap", "mean"),
            sd_ap=("ap", "std"),
            mean_train_ap=("train_ap", "mean"),
            mean_auroc=("auroc", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values("mean_ap", ascending=False)
    )

    return tuning_summary


def save_csv_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "fold_results.csv", index=False, mode="x")
    summary.to_csv(output_dir / "summary.csv", index=False, mode="x")



def build_metadata(
    configurations: list[dict],
    output_dir: Path,
) -> dict:
    metadata = {
        "feature_set": "extended",
        "imputed_categorical_features": [],
        "evaluation_partition": "train",
        "preprocessing": {
            "scale_numeric": False,
            "categorical_encoding": "onehot",
        },
        "cv": {
            "method": "GroupKFold",
            "group_column": "patient_nbr",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },
        "fixed_model_parameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": 8,
        },
        "configurations": configurations,
        "selection_metric": "mean_ap",
        "selection_direction": "maximize",
    }

    metadata["features"] = deepcopy(FEATURE_SETS["extended"])
    metadata["environment"] = {
        "python": platform.python_version(),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
    }

    paths = [
        Path("data/raw/diabetic_data.csv"),
        Path("data/processed/split_assignments.csv"),
        Path("data/processed/split_metadata.json"),
        output_dir / "fold_results.csv",
        output_dir / "summary.csv",
    ]

    metadata["file_sha256"] = {
        str(path): file_sha256(path)
        for path in paths
    }

    return metadata


def save_metadata(metadata: dict, output_dir: Path) -> None:
    with open(output_dir / "metadata.json", mode="x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")



def main(output_dir: Path) -> None:
    names = ["num_leaves", "min_child_samples", "reg_lambda"]
    combinations = list(product([15, 31], [20, 100], [0.0, 10.0]))

    configurations = [dict(zip(names, combo)) for combo in combinations]

    print(f"Configurations: {len(configurations)}")

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    train, _ = load_verified_partitions()
    results = run_search(train, configurations)
    summary = summarize_search(results)
    save_csv_results(results, summary, output_dir)
    metadata = build_metadata(configurations, output_dir)
    save_metadata(metadata, output_dir)

    print("Summary:\n", summary)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
