"""Check audit exports and failure paths without running a real bootstrap."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.audit_model import main


class AuditModelCommandTest(unittest.TestCase):
    def test_existing_directory_is_rejected_before_loading(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.audit_model.load_verified_partitions") as load:
                with self.assertRaises(FileExistsError):
                    main(Path(directory))
                load.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def run_synthetic_audit(self, destination, differences):
        validation = pd.DataFrame({
            "encounter_id": ["E1", "E2", "E3", "E4"],
            "patient_nbr": ["P1", "P1", "P2", "P2"],
            "age": ["[50-60)"] * 4,
            "gender": ["Female", "Female", "Male", "Male"],
            "race": ["?"] * 4,
            "readmitted": ["<30", "NO", "<30", ">30"],
        })
        scores = validation[["encounter_id", "patient_nbr"]].copy()
        scores["outcome"] = [1, 0, 1, 0]
        scores["catboost_specialty_mode"] = [0.30, 0.10, 0.40, 0.10]
        with patch("scripts.audit_model.load_verified_partitions", return_value=(None, validation)), patch(
            "scripts.audit_model.pd.read_csv", return_value=scores
        ) as read, patch(
            "scripts.audit_model.bootstrap_gender_recall_difference",
            return_value=np.asarray(differences),
        ) as bootstrap, patch(
            "scripts.audit_model.file_sha256", return_value="synthetic-hash"
        ), patch("builtins.print"):
            main(destination)
        self.assertEqual(read.call_args.kwargs["dtype"], {"encounter_id": str, "patient_nbr": str})
        self.assertEqual(bootstrap.call_args.kwargs, {"n_bootstrap": 1000, "random_state": 123})

    def test_valid_differences_export_summary_and_serializable_provenance(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "audit"
            self.run_synthetic_audit(destination, [-0.02, 0.01])
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertFalse(metadata["final_test_scored"])
            self.assertEqual(metadata["evaluation_partition"], "validation")
            self.assertEqual(metadata["bootstrap"]["finite_replicates"], 2)
            self.assertTrue(metadata["bootstrap"]["ci_reported"])
            self.assertIn("python", metadata["environment"])
            self.assertIn("carepath_lab/audit.py", metadata["input_sha256"])
            self.assertEqual(set(metadata["output_sha256"]), {
                "subgroup_metrics.csv", "subgroup_calibration.csv", "bootstrap_recall_summary.csv",
            })
            summary = pd.read_csv(destination / "bootstrap_recall_summary.csv").iloc[0]
            self.assertEqual(summary["observed_difference"], 0)
            self.assertAlmostEqual(summary["ci_lower"], -0.01925)
            self.assertAlmostEqual(summary["ci_upper"], 0.00925)
            self.assertEqual(summary["replicates"], 2)

    def test_undefined_differences_skip_ci_but_keep_tables_and_metadata(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "audit"
            self.run_synthetic_audit(destination, [-0.02, np.nan])
            self.assertFalse((destination / "bootstrap_recall_summary.csv").exists())
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(metadata["bootstrap"]["finite_replicates"], 1)
            self.assertFalse(metadata["bootstrap"]["ci_reported"])
            self.assertEqual(set(metadata["output_sha256"]), {
                "subgroup_metrics.csv", "subgroup_calibration.csv",
            })


if __name__ == "__main__":
    unittest.main()
