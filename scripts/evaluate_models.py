"""Compare fixed pipelines on saved validation and summarize calibration."""

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from carepath_lab.modeling import (
    build_baseline_model,
    build_random_forest_model,
    build_lightgbm_model,
    fit_and_predict_validation,
)
from carepath_lab.uncertainty import bootstrap_metric_scores
from scripts.train_baseline import load_verified_partitions, print_calibration_summary


def main() -> None:
    train, validation = load_verified_partitions()
    builders = {
        "logistic": build_baseline_model,
        "random_forest": build_random_forest_model,
        "lightgbm": build_lightgbm_model,
    }

    results = []
    predictions = {}
    for name, builder in builders.items():
        model, y_validation, probabilities = fit_and_predict_validation(
            train, validation, builder, feature_set="extended"
        )
        results.append(
            {
                "model": name,
                "encounters": len(y_validation),
                "ap": average_precision_score(y_validation, probabilities),
                "auroc": roc_auc_score(y_validation, probabilities),
                "brier": brier_score_loss(y_validation, probabilities),
            }
        )

        predictions[name] = probabilities

        print(f"\nCalibration - {name}")
        print_calibration_summary(
            y_validation,
            probabilities,
            validation["patient_nbr"],
        )

    print("\nSaved validation — fixed configurations, 45 predictors")
    print(
        pd.DataFrame(results).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    bootstrap_scores = bootstrap_metric_scores(
        y_validation,
        predictions,
        validation["patient_nbr"],
    )
    
    print("\nBootstrap Brier scores:", bootstrap_scores.shape)
    print(bootstrap_scores.head().to_string(index=False))

    comparisons = [
        ("lightgbm", "random_forest"),
        ("lightgbm", "logistic"),
        ("random_forest", "logistic"),
    ]

    metrics = pd.DataFrame(results).set_index("model")

    for first, second in comparisons:
        differences = (
            bootstrap_scores[first] - bootstrap_scores[second]
        )
        lower, upper = differences.quantile([0.025, 0.975])

        point_difference = (
            metrics.loc[first, "brier"]
            - metrics.loc[second, "brier"]
        )

        print(f"\nBrier difference ({first} - {second}): {point_difference:.6f}")
        print(
            "95% patient-cluster bootstrap percentile CI: "
            f"[{lower:.6f}, {upper:.6f}]"
        )

    bootstrap_ap_scores = bootstrap_metric_scores(
        y_validation,
        predictions,
        validation["patient_nbr"],
        score_function=average_precision_score,
    )

    print("\nBootstrap AP scores:", bootstrap_ap_scores.shape)
    print(bootstrap_ap_scores.head().to_string(index=False))

    
    differences_ap = (
        bootstrap_ap_scores["random_forest"] - bootstrap_ap_scores["lightgbm"]
    )
    lower_ap, upper_ap = differences_ap.quantile([0.025, 0.975])

    point_difference_ap = (
        metrics.loc["random_forest", "ap"] - metrics.loc["lightgbm", "ap"]
    )
    
    print(f"\nAP difference (RF - LightGBM): {point_difference_ap:.6f}")
    print(
        "95% patient-cluster bootstrap percentile CI: "
        f"[{lower_ap:.6f}, {upper_ap:.6f}]"
    )


if __name__ == "__main__":
    main()
