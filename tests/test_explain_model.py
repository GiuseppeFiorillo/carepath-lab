import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from carepath_lab.features import FEATURE_SETS
from scripts.explain_model import main


class ExplainModelCommandTest(unittest.TestCase):
    def test_existing_directory_is_rejected_before_loading_model_or_data(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.explain_model.joblib.load") as model, patch(
                "scripts.explain_model.load_verified_partitions"
            ) as data:
                with self.assertRaises(FileExistsError):
                    main(Path(directory))
                model.assert_not_called()
                data.assert_not_called()

    def run_synthetic_export(self, destination, changed_inputs=False, wrong_model=False, wrong_split=False):
        size = 1002
        config = FEATURE_SETS["reviewed"]
        validation = pd.DataFrame({
            **{name: list(range(size)) for name in config["numeric"]},
            **{name: ["None"] * size for name in config["categorical"]},
            "encounter_id": [f"E{i}" for i in range(size)],
            "patient_nbr": [f"P{i // 2}" for i in range(size)],
            "readmitted": ["NO"] * size,
        }, index=range(10000, 10000 + 2 * size, 2))
        source_calls = 0
        source_names = [
            "data/raw/diabetic_data.csv", "data/processed/split_assignments.csv",
            "data/processed/split_metadata.json",
        ]
        model_metadata = {
            "feature_set": "reviewed",
            "input_sha256": {name: "synthetic-stable-hash" for name in source_names},
            "output_sha256": {"selected_pipeline.joblib": "wrong" if wrong_model else "synthetic-stable-hash"},
        }
        if wrong_split:
            model_metadata["input_sha256"]["data/processed/split_assignments.csv"] = "wrong"
        original_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == Path("artifacts/current_model/metadata.json"):
                return json.dumps(model_metadata)
            return original_read_text(path, *args, **kwargs)

        def fingerprint(path):
            nonlocal source_calls
            if path == Path("data/raw/diabetic_data.csv"):
                source_calls += 1
                if changed_inputs and source_calls > 1:
                    return "changed"
            return "synthetic-stable-hash"

        def explain(model, inputs):
            # Synthetic helper output checks I/O alignment, not SHAP mathematics.
            values = np.zeros((len(inputs), len(inputs.columns) + 1))
            values[:, 0] = inputs["number_inpatient"].to_numpy()
            values[:, -1] = -2.0
            return inputs.copy(), values

        with patch("scripts.explain_model.joblib.load", return_value=object()), patch(
            "scripts.explain_model.load_verified_partitions", return_value=(None, validation)
        ), patch("scripts.explain_model.compute_native_shap", side_effect=explain), patch(
            "scripts.explain_model.file_sha256", side_effect=fingerprint
        ), patch.object(Path, "read_text", autospec=True, side_effect=read_text
        ), patch("builtins.print"):
            main(destination)

    def test_export_preserves_sample_identity_order_arrays_and_provenance(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            self.run_synthetic_export(destination)
            sample = pd.read_csv(destination / "sample.csv", dtype=str)
            inputs = pd.read_csv(destination / "transformed_inputs.csv", keep_default_na=False)
            with np.load(destination / "shap_values.npz", allow_pickle=False) as archive:
                values = archive["contributions"]
                references = archive["reference_values"]
                names = archive["feature_names"].tolist()
            source_rows = sample.encounter_id.str.removeprefix("E").astype(int)
            np.testing.assert_array_equal(inputs.number_inpatient, source_rows)
            np.testing.assert_array_equal(values[:, 0], source_rows)
            np.testing.assert_array_equal(references, np.full(1000, -2.0))
            self.assertEqual(names, list(inputs.columns))
            self.assertEqual(sample.shape, (1000, 2))
            self.assertEqual(values.shape, (1000, 41))
            self.assertTrue(sample.encounter_id.is_unique)
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(metadata["sampling"], {"n": 1000, "random_state": 42, "unit": "encounter"})
            self.assertFalse(metadata["final_test_scored"])
            self.assertEqual(metadata["shap"]["output_units"], "raw_log_odds")
            self.assertEqual(set(metadata["output_sha256"]), {
                "sample.csv", "transformed_inputs.csv", "shap_values.npz",
                "global_importance.png", "beeswarm.png", "waterfall.png",
            })
            self.assertEqual(metadata["waterfall"], {
                "sample_row": 0,
                "selection": "first row of the fixed random validation sample",
            })
            for filename in ["global_importance.png", "beeswarm.png", "waterfall.png"]:
                self.assertTrue((destination / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("matplotlib", metadata["environment"])
            self.assertIn("shap", metadata["environment"])
            self.assertIn("python", metadata["environment"])
            self.assertTrue(all(isinstance(value, str) for value in metadata["environment"].values()))

    def test_changed_inputs_prevent_completed_metadata(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "Inputs or code"):
                self.run_synthetic_export(destination, changed_inputs=True)
            self.assertFalse((destination / "metadata.json").exists())

    def test_mismatched_model_fingerprint_is_rejected(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "model fingerprint"):
                self.run_synthetic_export(destination, wrong_model=True)
            self.assertFalse((destination / "metadata.json").exists())

    def test_split_disagreeing_with_model_training_is_rejected(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            with self.assertRaisesRegex(ValueError, "Source or split"):
                self.run_synthetic_export(destination, wrong_split=True)
            self.assertFalse((destination / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
