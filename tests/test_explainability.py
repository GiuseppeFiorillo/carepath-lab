import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import Pool
import numpy as np
import pandas as pd
from scipy.special import expit

from carepath_lab.explainability import compute_native_shap, plot_beeswarm, plot_global_importance, plot_waterfall
from carepath_lab.features import FEATURE_SETS
from carepath_lab.modeling import build_catboost_model


class ComputeNativeShapTest(unittest.TestCase):
    def setUp(self):
        config = FEATURE_SETS["reviewed"]
        self.inputs = pd.DataFrame({
            **{name: list(range(1, 9)) for name in config["numeric"]},
            **{name: ["A", "B"] * 4 for name in config["categorical"]},
            "medical_specialty": ["Cardiology", "?", "Cardiology", "Other"] * 2,
        }, index=range(10, 90, 10))
        self.pipeline = build_catboost_model(
            feature_set="reviewed", n_jobs=1,
            imputed_categorical_features=("medical_specialty",),
        )
        self.pipeline.set_params(classifier__iterations=8, classifier__depth=2)
        self.pipeline.fit(self.inputs, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_returns_aligned_contributions_reconstructing_scores_and_probabilities(self):
        transformed, values = compute_native_shap(self.pipeline, self.inputs)
        classifier = self.pipeline.named_steps["classifier"]
        self.assertEqual(values.shape, (8, 42))
        self.assertEqual(list(transformed.columns), classifier.feature_names_)
        self.assertTrue(transformed.index.equals(self.inputs.index))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(transformed["medical_specialty"].iloc[1], "Cardiology")
        pool = Pool(transformed, cat_features=classifier.get_cat_feature_indices())
        raw = classifier.predict(pool, prediction_type="RawFormulaVal")
        self.assertGreater(np.std(raw), 0)
        np.testing.assert_allclose(values.sum(axis=1), raw, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            expit(values.sum(axis=1)), self.pipeline.predict_proba(self.inputs)[:, 1],
            rtol=1e-6, atol=1e-8,
        )

    def test_calls_preserve_inputs_and_fitted_model_and_repeat_identically(self):
        before = self.inputs.copy(deep=True)
        classifier = self.pipeline.named_steps["classifier"]
        leaves = classifier.get_leaf_values().copy()
        imputer = self.pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical_imputed"
        ].named_steps["imputer"]
        statistics = imputer.statistics_.copy()
        first_inputs, first = compute_native_shap(self.pipeline, self.inputs)
        repeated_inputs, repeated = compute_native_shap(self.pipeline, self.inputs)
        pd.testing.assert_frame_equal(self.inputs, before)
        pd.testing.assert_frame_equal(first_inputs, repeated_inputs)
        np.testing.assert_allclose(first, repeated)
        np.testing.assert_array_equal(leaves, classifier.get_leaf_values())
        np.testing.assert_array_equal(statistics, imputer.statistics_)

    def test_additivity_guard_rejects_corrupted_native_contributions(self):
        _, values = compute_native_shap(self.pipeline, self.inputs)
        corrupted = values.copy()
        corrupted[0, -1] += 1
        classifier = self.pipeline.named_steps["classifier"]
        with patch.object(classifier, "get_feature_importance", return_value=corrupted):
            with self.assertRaises(AssertionError):
                compute_native_shap(self.pipeline, self.inputs)

    def test_reordered_transformed_columns_are_rejected(self):
        preprocessor = self.pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(self.inputs)
        reordered = transformed.iloc[:, ::-1]
        with patch.object(preprocessor, "transform", return_value=reordered):
            with self.assertRaisesRegex(ValueError, "feature order"):
                compute_native_shap(self.pipeline, self.inputs)

    def test_extra_zero_column_missing_reference_or_missing_row_are_rejected(self):
        _, values = compute_native_shap(self.pipeline, self.inputs)
        malformed = {
            "extra_zero_column": np.column_stack([values, np.zeros(len(values))]),
            "missing_reference": values[:, :-1],
            "missing_row": values[:-1],
        }
        classifier = self.pipeline.named_steps["classifier"]
        for name, matrix in malformed.items():
            with self.subTest(case=name), patch.object(
                classifier, "get_feature_importance", return_value=matrix
            ):
                with self.assertRaisesRegex(ValueError, "shape"):
                    compute_native_shap(self.pipeline, self.inputs)

    def test_nan_and_infinite_contributions_are_rejected(self):
        _, values = compute_native_shap(self.pipeline, self.inputs)
        classifier = self.pipeline.named_steps["classifier"]
        for value in [np.nan, np.inf, -np.inf]:
            corrupted = values.copy()
            corrupted[0, 0] = value
            with self.subTest(value=value), patch.object(
                classifier, "get_feature_importance", return_value=corrupted
            ):
                with self.assertRaisesRegex(ValueError, "finite"):
                    compute_native_shap(self.pipeline, self.inputs)

    def test_probability_reconstruction_guard_rejects_inconsistent_probabilities(self):
        probabilities = self.pipeline.predict_proba(self.inputs).copy()
        probabilities[0] = [0.01, 0.99]
        with patch.object(self.pipeline, "predict_proba", return_value=probabilities):
            with self.assertRaises(AssertionError):
                compute_native_shap(self.pipeline, self.inputs)


class PlotGlobalImportanceTest(unittest.TestCase):
    def test_bar_widths_use_absolute_means_with_matching_labels_and_sample_count(self):
        values = np.array([[1.0, -2.0, 0.0], [-3.0, 0.0, 0.0]])
        before = values.copy()
        figure = plot_global_importance(values, ["a", "b", "c"])
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], ["c", "b", "a"])
        np.testing.assert_allclose([bar.get_width() for bar in axis.patches], [0.0, 1.0, 2.0])
        self.assertIn("log-odds", axis.get_xlabel())
        self.assertEqual(axis.get_title(), "CatBoost — 2 validation encounters")
        np.testing.assert_array_equal(values, before)

    def test_only_ten_largest_features_are_displayed(self):
        values = np.tile(np.arange(12, dtype=float), (2, 1))
        figure = plot_global_importance(values, [f"f{i}" for i in range(12)])
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]
        self.assertEqual(len(axis.patches), 10)
        self.assertEqual([label.get_text() for label in axis.get_yticklabels()], [f"f{i}" for i in range(2, 12)])
        np.testing.assert_allclose([bar.get_width() for bar in axis.patches], np.arange(2, 12))


class PlotBeeswarmTest(unittest.TestCase):
    def test_returns_signed_contributions_without_adding_reference(self):
        values = np.array([[-0.6, 0.1], [0.0, 0.2], [0.6, -0.1]])
        references = np.full(3, 10.0)
        before = values.copy()
        figure = plot_beeswarm(values, references, ["a", "b"])
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]
        plotted_x = np.concatenate([
            collection.get_offsets()[:, 0]
            for collection in axis.collections
            if len(collection.get_offsets())
        ])
        np.testing.assert_allclose(np.sort(plotted_x), np.sort(values.ravel()))
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual({label.get_text() for label in axis.get_yticklabels()}, {"a", "b"})
        self.assertIn("log-odds", axis.get_xlabel())
        self.assertEqual(axis.get_title(), "CatBoost — 3 validation encounters")
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(references, np.full(3, 10.0))


class PlotWaterfallTest(unittest.TestCase):
    def test_preserves_local_values_reference_and_feature_labels(self):
        import shap

        values = np.array([-0.6, 0.2, 0.1])
        features = np.array(["0", "Cardiology", "250.8"])
        before = values.copy()
        with patch.object(shap.plots, "waterfall", wraps=shap.plots.waterfall) as plot:
            figure = plot_waterfall(values, -2.0, features, ["count", "specialty", "diagnosis"])
        self.addCleanup(plt.close, figure)
        explanation = plot.call_args.args[0]
        np.testing.assert_array_equal(explanation.values, values)
        np.testing.assert_array_equal(explanation.data, features)
        self.assertEqual(explanation.base_values, -2.0)
        self.assertEqual(explanation.feature_names, ["count", "specialty", "diagnosis"])
        self.assertAlmostEqual(explanation.base_values + explanation.values.sum(), -2.3)
        self.assertFalse(plot.call_args.kwargs["show"])
        self.assertTrue(any(axis.get_xlabel() == "Model score (log-odds)" for axis in figure.axes))
        self.assertTrue(any("Cardiology" in label.get_text() for label in figure.axes[0].get_yticklabels()))
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
