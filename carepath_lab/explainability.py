import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit
from sklearn.pipeline import Pipeline


def compute_native_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return transformed inputs and native SHAP values in raw-score units."""

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_inputs = preprocessor.transform(X)

    if list(transformed_inputs.columns) != classifier.feature_names_:
        raise ValueError(
            "Transformed feature order must match classifier feature names."
        )

    data_pool = Pool(
        transformed_inputs,
        cat_features=classifier.get_cat_feature_indices(),
    )

    shap_values = classifier.get_feature_importance(
        data=data_pool,
        type="ShapValues",
        thread_count=8,
    )

    expected_shape = (len(X), len(classifier.feature_names_) + 1)

    if shap_values.shape != expected_shape:
        raise ValueError("Unexpected SHAP matrix shape.")

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values must all be finite.")

    feature_contributions = shap_values[:, :-1]
    reference_values = shap_values[:, -1]

    reconstructed_scores = (
        reference_values + feature_contributions.sum(axis=1)
    )
    raw_scores = classifier.predict(
        data_pool,
        prediction_type="RawFormulaVal"
    )

    np.testing.assert_allclose(
        reconstructed_scores,
        raw_scores,
        rtol=1e-6,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        expit(reconstructed_scores),
        pipeline.predict_proba(X)[:, 1],
        rtol=1e-6,
        atol=1e-8,
    )

    return transformed_inputs, shap_values


def plot_global_importance(
    contributions: np.ndarray,
    feature_names: list[str],
):
    """Return a bar plot of the ten largest mean absolute SHAP contributions."""

    import matplotlib.pyplot as plt

    importance = pd.Series(
        np.abs(contributions).mean(axis=0),
        index=feature_names,
    )

    top_features = importance.nlargest(10).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    top_features.plot.barh(ax=ax, color="steelblue")
    ax.set(
        xlabel="Mean absolute SHAP contribution (log-odds)",
        ylabel="Feature",
        title=f"CatBoost — {len(contributions):,} validation encounters"
    )
    fig.tight_layout()

    return fig


def plot_beeswarm(
    contributions: np.ndarray,
    reference_values: np.ndarray,
    feature_names: list[str],
):
    """Return a monochrome beeswarm of the ten leading SHAP features."""

    import shap

    explanation = shap.Explanation(
        values=contributions,
        base_values=reference_values,
        feature_names=feature_names,
    )

    ax = shap.plots.beeswarm(
        explanation,
        max_display=10,
        group_remaining_features=False,
        color="steelblue",
        color_bar=False,
        show=False,
    )

    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title(f"CatBoost — {len(contributions):,} validation encounters")
    ax.figure.tight_layout()

    return ax.figure


def plot_waterfall(
    contributions: np.ndarray,
    reference_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
):
    """Return a waterfall explaining one encounter in raw log-odds."""

    import shap

    local_explanation = shap.Explanation(
        values=contributions,
        base_values=reference_value,
        data=feature_values,
        feature_names=feature_names,
    )

    ax = shap.plots.waterfall(
        local_explanation,
        max_display=10,
        show=False,
    )
    ax.set_xlabel("Model score (log-odds)")

    return ax.figure
