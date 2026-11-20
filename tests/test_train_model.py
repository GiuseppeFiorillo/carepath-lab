import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from carepath_lab.features import FEATURE_SETS
from carepath_lab.modeling import build_catboost_model
from scripts.train_model import main


class TrainModelCommandTest(unittest.TestCase):
    def test_existing_destination_is_refused_before_loading_or_fitting(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.train_model.load_verified_partitions") as load:
                with self.assertRaises(FileExistsError):
                    main(Path(directory))
                load.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def run_synthetic_training(self, destination, change_inputs=False):
        config = FEATURE_SETS["reviewed"]
        train = pd.DataFrame({
            **{name: list(range(1, 9)) for name in config["numeric"]},
            **{name: ["A", "B"] * 4 for name in config["categorical"]},
            "medical_specialty": ["Cardiology", "?", "Cardiology", "Other"] * 2,
            "encounter_id": [f"T{i}" for i in range(8)],
            "patient_nbr": [f"TRAIN_{i}" for i in range(8)],
            "readmitted": ["NO"] * 4 + ["<30"] * 4,
        })
        validation = train.iloc[:5].copy()
        validation["encounter_id"] = [f"V{i}" for i in range(5)]
        validation["patient_nbr"] = [f"VALIDATION_{i}" for i in range(5)]
        decision = json.loads(Path("docs/reports/feature_review_selection.json").read_text())
        raw_hash_calls = 0

        def fingerprint(path):
            nonlocal raw_hash_calls
            if path == Path("docs/reports/feature_review_folds.csv"):
                return decision["fold_results_sha256"]
            if path == Path("data/raw/diabetic_data.csv"):
                raw_hash_calls += 1
                if change_inputs and raw_hash_calls > 1:
                    return "changed"
            return "synthetic-stable-hash"

        def tiny_builder(**kwargs):
            kwargs["n_jobs"] = 1
            model = build_catboost_model(**kwargs)
            model.set_params(classifier__iterations=3, classifier__depth=2)
            return model

        with patch("scripts.train_model.load_verified_partitions", return_value=(train, validation)), patch(
            "scripts.train_model.file_sha256", side_effect=fingerprint
        ), patch("scripts.train_model.build_catboost_model", side_effect=tiny_builder), patch("builtins.print"):
            main(destination)

    def test_current_training_exports_checked_model_predictions_and_metadata(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            self.run_synthetic_training(destination)
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(metadata["training_encounters"], 8)
            self.assertEqual(metadata["validation_encounters"], 5)
            self.assertEqual(metadata["feature_set"], "reviewed")
            self.assertFalse(metadata["final_test_scored"])
            self.assertLess(metadata["shap_smoke_check"]["maximum_additivity_error"], 1e-8)
            predictions = pd.read_csv(destination / "validation_predictions.csv")
            self.assertEqual(predictions.encounter_id.tolist(), [f"V{i}" for i in range(5)])
            self.assertTrue((destination / "selected_pipeline.joblib").exists())

    def test_changed_inputs_during_training_reject_completion_metadata(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "Inputs or code"):
                self.run_synthetic_training(destination, change_inputs=True)
            self.assertFalse((destination / "metadata.json").exists())

    def test_changed_cv_evidence_is_rejected_before_loading_or_fitting(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.train_model.load_verified_partitions") as load, patch(
                "scripts.train_model.file_sha256", return_value="changed-hash"
            ):
                load.side_effect = AssertionError("Training must not start with changed CV evidence")
                with self.assertRaisesRegex(ValueError, "CV evidence"):
                    main(Path(directory) / "new")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
