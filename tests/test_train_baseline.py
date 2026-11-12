import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from carepath_lab.modeling import build_baseline_model
from scripts.train_baseline import main, print_calibration_summary


class TrainingInputCommandTest(unittest.TestCase):
    def test_calibration_summary_preserves_index_alignment_and_patient_support(self):
        outcomes = pd.Series([1, 0, 1], index=[30, 10, 20])
        probabilities = np.array([0.05, 0.08, 0.09])
        patients = pd.Series(["B", "A", "A"], index=[20, 30, 10])
        original_outcomes = outcomes.copy()
        original_patients = patients.copy()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_calibration_summary(outcomes, probabilities, patients)

        # One bin: three encounters, two positives and two distinct patients.
        self.assertRegex(output.getvalue(), r"3\s+0\.073333\s+0\.666667\s+2\s+2")
        pd.testing.assert_series_equal(outcomes, original_outcomes)
        pd.testing.assert_series_equal(patients, original_patients)
        np.testing.assert_array_equal(probabilities, [0.05, 0.08, 0.09])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source_path = self.root / "data/raw/diabetic_data.csv"
        self.assignment_path = self.root / "data/processed/split_assignments.csv"
        self.metadata_path = self.root / "data/processed/split_metadata.json"
        self.source_path.parent.mkdir(parents=True)
        self.assignment_path.parent.mkdir(parents=True)
        data = pd.DataFrame(
            {
                "encounter_id": [f"E{i}" for i in range(7)],
                "patient_nbr": [f"P{i}" for i in range(7)],
                "discharge_disposition_id": ["1"] * 7,
                "time_in_hospital": ["1", "2", "3", "4", "14", "13", "12"],
                "num_medications": ["3"] * 7,
                "number_emergency": ["0"] * 7,
                "number_inpatient": ["0"] * 7,
                "age": ["[20-30)"] * 7,
                "admission_type_id": ["1"] * 7,
                "readmitted": ["<30", "NO", ">30", "<30", "NO", "<30", "NO"],
            }
        )
        data.to_csv(self.source_path, index=False)
        assignments = data[["encounter_id", "patient_nbr"]].copy()
        assignments["partition"] = ["train"] * 4 + ["validation"] * 2 + ["test"]
        assignments.to_csv(self.assignment_path, index=False)
        self.metadata = {
            "source": {"sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest()}
        }
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_prepares_training_validation_inputs_without_writing_artifacts(self):
        paths = [self.source_path, self.assignment_path, self.metadata_path]
        before = [path.read_bytes() for path in paths]
        output = io.StringIO()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(output):
            main(n_bootstrap=2)
        self.assertIn("Training: X (4, 6), y (4,)", output.getvalue())
        self.assertIn("Validation: X (2, 6), y (2,)", output.getvalue())
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_changed_source_hash_is_rejected_before_input_construction(self):
        self.metadata["source"]["sha256"] = "0" * 64
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        with contextlib.chdir(self.root), patch(
            "scripts.train_baseline.build_model_inputs"
        ) as builder:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                main(n_bootstrap=2)
            builder.assert_not_called()

    def test_pipeline_fits_and_predicts_using_only_designated_partitions(self):
        model = build_baseline_model()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            with patch("scripts.train_baseline.build_baseline_model", return_value=model):
                with patch.object(model, "fit", wraps=model.fit) as fit, patch.object(
                    model, "predict_proba", wraps=model.predict_proba
                ) as predict:
                    main(n_bootstrap=2)

        fit.assert_called_once()
        X_train, y_train = fit.call_args.args
        self.assertEqual(X_train["time_in_hospital"].tolist(), [1, 2, 3, 4])
        self.assertEqual(y_train.tolist(), [1, 0, 0, 1])
        scaler = model.named_steps["preprocessor"].named_transformers_["numeric"]
        self.assertEqual(scaler.mean_[0], 2.5)
        self.assertEqual(model.classes_.tolist(), [0, 1])
        predict.assert_called_once()
        X_validation = predict.call_args.args[0]
        self.assertEqual(X_validation["time_in_hospital"].tolist(), [14, 13])

    def test_dummy_uses_training_prior_and_predicts_only_validation(self):
        dummy = DummyClassifier(strategy="prior")
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            with patch("scripts.train_baseline.DummyClassifier", return_value=dummy):
                with patch.object(dummy, "fit", wraps=dummy.fit) as fit, patch.object(
                    dummy, "predict_proba", wraps=dummy.predict_proba
                ) as predict:
                    main(n_bootstrap=2)

        fit.assert_called_once()
        X_train, y_train = fit.call_args.args
        self.assertEqual(X_train["time_in_hospital"].tolist(), [1, 2, 3, 4])
        self.assertEqual(y_train.tolist(), [1, 0, 0, 1])
        self.assertEqual(dummy.classes_.tolist(), [0, 1])
        self.assertEqual(dummy.class_prior_[1], 0.5)
        predict.assert_called_once()
        self.assertEqual(
            predict.call_args.args[0]["time_in_hospital"].tolist(), [14, 13]
        )

    def test_metrics_use_validation_targets_and_positive_class_probabilities(self):
        model = build_baseline_model()
        output = io.StringIO()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(output):
            with patch("scripts.train_baseline.build_baseline_model", return_value=model):
                with patch.object(
                    model, "predict_proba", return_value=np.array([[0.2, 0.8], [0.4, 0.6]])
                ), patch(
                    "scripts.train_baseline.brier_score_loss", wraps=brier_score_loss
                ) as metric, patch(
                    "scripts.train_baseline.roc_auc_score", wraps=roc_auc_score
                ) as auroc, patch(
                    "scripts.train_baseline.average_precision_score",
                    wraps=average_precision_score,
                ) as ap, patch(
                    "scripts.train_baseline.precision_recall_curve",
                    wraps=precision_recall_curve,
                ) as pr_curve:
                    main(n_bootstrap=2)

        self.assertEqual(metric.call_count, 6)
        for call in metric.call_args_list[:2]:
            self.assertEqual(call.args[0].tolist(), [0, 1])
        np.testing.assert_allclose(metric.call_args_list[0].args[1], [0.8, 0.6])
        np.testing.assert_allclose(metric.call_args_list[1].args[1], [0.5, 0.5])
        self.assertIn("Validation Brier score (logistic): 0.400000", output.getvalue())
        self.assertIn("Validation Brier score (dummy): 0.250000", output.getvalue())
        self.assertEqual(auroc.call_count, 2)
        for call in auroc.call_args_list:
            self.assertEqual(call.args[0].tolist(), [0, 1])
        np.testing.assert_allclose(auroc.call_args_list[0].args[1], [0.8, 0.6])
        np.testing.assert_allclose(auroc.call_args_list[1].args[1], [0.5, 0.5])
        self.assertIn("Validation AUROC score (logistic): 0.000000", output.getvalue())
        self.assertIn("Validation AUROC score (dummy): 0.500000", output.getvalue())
        self.assertEqual(ap.call_count, 2)
        for call in ap.call_args_list:
            self.assertEqual(call.args[0].tolist(), [0, 1])
        np.testing.assert_allclose(ap.call_args_list[0].args[1], [0.8, 0.6])
        np.testing.assert_allclose(ap.call_args_list[1].args[1], [0.5, 0.5])
        self.assertIn("Validation AP score (logistic): 0.500000", output.getvalue())
        self.assertIn("Validation AP score (dummy): 0.500000", output.getvalue())
        # For this reversed ranking, trapezoidal PR area is 0.25, versus AP 0.5.
        self.assertIn(
            "Validation PR-AUC (logistic, trapezoidal): 0.250000", output.getvalue()
        )
        self.assertIn("Validation precision (threshold=0.50): 50.00%", output.getvalue())
        self.assertIn("Validation recall (threshold=0.50): 100.00%", output.getvalue())
        pr_curve.assert_called_once()
        self.assertEqual(pr_curve.call_args.args[0].tolist(), [0, 1])
        np.testing.assert_allclose(pr_curve.call_args.args[1], [0.8, 0.6])
        self.assertIn("precision=(3,), recall=(3,), thresholds=(2,)", output.getvalue())
        self.assertIn("First pair (precision, recall): (0.500000, 1.000000)", output.getvalue())
        self.assertIn("Last pair (precision, recall): (1.000000, 0.000000)", output.getvalue())

    def test_confusion_matrices_use_inclusive_thresholds_and_fixed_label_order(self):
        model = build_baseline_model()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            with patch("scripts.train_baseline.build_baseline_model", return_value=model):
                with patch.object(
                    model, "predict_proba", return_value=np.array([[0.5, 0.5], [0.8, 0.2]])
                ), patch(
                    "scripts.train_baseline.confusion_matrix", wraps=confusion_matrix
                ) as metric:
                    main(n_bootstrap=2)

        self.assertEqual(metric.call_count, 2)
        for call, expected in zip(metric.call_args_list, [[1, 0], [1, 1]]):
            actual, predicted = call.args
            self.assertEqual(actual.tolist(), [0, 1])
            self.assertEqual(predicted.tolist(), expected)
            self.assertEqual(call.kwargs["labels"], [0, 1])

    def test_bootstrap_reuses_same_positions_with_duplicate_draws(self):
        # An unbalanced prior makes the reference loss depend on sampled outcomes.
        data = pd.read_csv(self.source_path, dtype=str, keep_default_na=False)
        data.loc[3, "readmitted"] = "NO"
        data.to_csv(self.source_path, index=False)
        self.metadata["source"]["sha256"] = hashlib.sha256(
            self.source_path.read_bytes()
        ).hexdigest()
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        model = build_baseline_model()
        output = io.StringIO()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(output):
            with patch("scripts.train_baseline.build_baseline_model", return_value=model):
                with patch.object(
                    model, "predict_proba", return_value=np.array([[0.2, 0.8], [0.4, 0.6]])
                ), patch(
                    "scripts.train_baseline.sample_patient_positions",
                    return_value=[1, 0, 1],
                ) as sample, patch(
                    "scripts.train_baseline.brier_score_loss", wraps=brier_score_loss
                ) as metric, patch(
                    "scripts.train_baseline.np.quantile", wraps=np.quantile
                ) as quantiles:
                    main(n_bootstrap=2)

        self.assertEqual(sample.call_count, 2)
        self.assertIs(sample.call_args_list[0].args[1], sample.call_args_list[1].args[1])
        self.assertEqual(metric.call_count, 6)
        for offset in [2, 4]:
            logistic_call, dummy_call = metric.call_args_list[offset:offset + 2]
            self.assertIs(logistic_call.args[0], dummy_call.args[0])
            self.assertEqual(logistic_call.args[0].tolist(), [1, 0, 1])
            np.testing.assert_allclose(logistic_call.args[1], [0.6, 0.8, 0.6])
            np.testing.assert_allclose(dummy_call.args[1], [0.25, 0.25, 0.25])
        self.assertIn(
            "Bootstrap replicate Brier mean: 0.320000",
            output.getvalue(),
        )
        self.assertIn("Bootstrap Brier summary (2 patient-cluster replicates)", output.getvalue())
        self.assertEqual(quantiles.call_count, 2)
        np.testing.assert_allclose(quantiles.call_args_list[0].args[0], [0.32, 0.32])
        expected_difference = 0.32 - (0.75**2 + 0.25**2 + 0.75**2) / 3
        np.testing.assert_allclose(
            quantiles.call_args_list[1].args[0], [expected_difference] * 2
        )
        for call in quantiles.call_args_list:
            np.testing.assert_allclose(call.args[1], [0.025, 0.975])
        self.assertIn(
            "95% patient-cluster bootstrap percentile CI: [0.320000, 0.320000]",
            output.getvalue(),
        )
        self.assertIn("Original logistic-dummy Brier difference: 0.087500", output.getvalue())
        self.assertIn("[-0.075833, -0.075833]", output.getvalue())
