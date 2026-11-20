import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from carepath_lab.splitting import (
    make_patient_folds,
    split_patient_holdout,
    validate_split_assignments,
)
from scripts.split_data import main


class VerifyTestSplit(unittest.TestCase):
    def test_no_patient_overlap(self):
        data = pd.DataFrame(
            {
                "encounter_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
                "patient_nbr": ["P1", "P1", "P2", "P3", "P4", "P4"],
            },
            index=[10, 20, 30, 40, 50, 60],
        )

        development, test = split_patient_holdout(
            data, test_size=0.5, random_state=42
        )
        development_patients = set(development["patient_nbr"])
        test_patients = set(test["patient_nbr"])

        intersection = development_patients & test_patients
        self.assertEqual(intersection, set())

        development_encounters = set(development["encounter_id"])
        test_encounters = set(test["encounter_id"])
        union = development_encounters | test_encounters
        self.assertEqual(union, set(data["encounter_id"]))
        self.assertEqual(len(development) + len(test), len(data))

        development_again, test_again = split_patient_holdout(
            data, test_size=0.5, random_state=42
        )

        pd.testing.assert_frame_equal(development, development_again)
        pd.testing.assert_frame_equal(test, test_again)

        train, validation = split_patient_holdout(
            development, test_size=0.5, random_state=42
        )

        train_patients = set(train["patient_nbr"])
        validation_patients = set(validation["patient_nbr"])

        self.assertEqual(train_patients & validation_patients, set())
        self.assertEqual(train_patients & test_patients, set())
        self.assertEqual(validation_patients & test_patients, set())


class ValidateSplitAssignmentsTest(unittest.TestCase):
    def setUp(self):
        self.cohort = pd.DataFrame(
            {
                "encounter_id": ["E1", "E2", "E3", "E4"],
                "patient_nbr": ["P1", "P1", "P2", "P3"],
            }
        )
        self.assignments = self.cohort.copy()
        self.assignments["partition"] = ["train", "train", "validation", "test"]

    def test_valid_assignments_allow_repeated_patients_in_same_partition(self):
        result = validate_split_assignments(self.assignments, self.cohort)
        self.assertIsNone(result)

    def test_duplicate_encounter_is_rejected(self):
        duplicated = pd.concat(
            [self.assignments, self.assignments.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "Duplicate encounter IDs"):
            validate_split_assignments(duplicated, self.cohort)

    def test_swapped_patients_are_rejected_despite_matching_id_sets(self):
        swapped = self.assignments.copy()
        swapped.loc[2, "patient_nbr"] = "P3"
        swapped.loc[3, "patient_nbr"] = "P2"
        self.assertEqual(set(swapped["patient_nbr"]), set(self.cohort["patient_nbr"]))
        self.assertEqual(set(swapped["encounter_id"]), set(self.cohort["encounter_id"]))
        with self.assertRaisesRegex(ValueError, "encounter/patient pairs"):
            validate_split_assignments(swapped, self.cohort)

    def test_valid_assignments_in_different_row_order_are_accepted(self):
        reordered = self.assignments.iloc[::-1].reset_index(drop=True)
        result = validate_split_assignments(reordered, self.cohort)
        self.assertIsNone(result)

    def test_unexpected_partition_label_is_rejected(self):
        invalid = self.assignments.copy()
        invalid.loc[0, "partition"] = "trian"
        with self.assertRaisesRegex(ValueError, "Unexpected partition categories.*trian"):
            validate_split_assignments(invalid, self.cohort)

    def test_patient_in_multiple_partitions_is_rejected(self):
        overlapping = self.assignments.copy()
        overlapping.loc[1, "partition"] = "validation"
        with self.assertRaisesRegex(ValueError, "Each patient"):
            validate_split_assignments(overlapping, self.cohort)

    def test_missing_or_unexpected_encounters_are_rejected(self):
        missing = self.assignments.iloc[1:].copy()
        extra = self.assignments.iloc[[0]].copy()
        extra["encounter_id"] = "E5"
        extra["patient_nbr"] = "P4"
        unexpected = pd.concat([self.assignments, extra], ignore_index=True)
        replaced = self.assignments.copy()
        replaced.loc[0, "encounter_id"] = "E5"

        for case in [missing, unexpected, replaced]:
            with self.subTest(encounters=case["encounter_id"].tolist()):
                with self.assertRaisesRegex(ValueError, "do not match the cohort"):
                    validate_split_assignments(case, self.cohort)


class SplitCommandTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        raw_directory = self.root / "data/raw"
        raw_directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "encounter_id": [f"E{i}" for i in range(8)],
                "patient_nbr": [f"P{i}" for i in range(8)],
                "discharge_disposition_id": ["1"] * 8,
                "readmitted": ["<30", "NO"] * 4,
            }
        ).to_csv(raw_directory / "diabetic_data.csv", index=False)
        self.output_path = self.root / "data/processed/split_assignments.csv"
        self.metadata_path = self.root / "data/processed/split_metadata.json"

    def run_command(self):
        output = io.StringIO()
        with contextlib.chdir(self.root), contextlib.redirect_stdout(output):
            main()
        return output.getvalue()

    def test_creates_then_reuses_matching_assignments_without_rewriting(self):
        self.run_command()
        before = self.output_path.read_bytes()
        before_mtime = self.output_path.stat().st_mtime_ns
        metadata_before = self.metadata_path.read_bytes()
        metadata_mtime = self.metadata_path.stat().st_mtime_ns
        metadata = json.loads(metadata_before)
        self.assertEqual(metadata["cohort"]["encounter_count"], 8)
        self.assertEqual(metadata["split"]["random_state"], 42)
        self.assertEqual(
            metadata["source"]["sha256"],
            hashlib.sha256((self.root / "data/raw/diabetic_data.csv").read_bytes()).hexdigest(),
        )
        output = self.run_command()
        self.assertIn("match; reusing them.", output)
        self.assertEqual(self.output_path.read_bytes(), before)
        self.assertEqual(self.output_path.stat().st_mtime_ns, before_mtime)
        self.assertEqual(self.metadata_path.read_bytes(), metadata_before)
        self.assertEqual(self.metadata_path.stat().st_mtime_ns, metadata_mtime)

    def test_equivalent_metadata_formatting_is_reused(self):
        self.run_command()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.metadata_path.write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        before = self.metadata_path.read_bytes()
        output = self.run_command()
        self.assertIn("Existing metadata", output)
        self.assertEqual(self.metadata_path.read_bytes(), before)

    def test_environment_change_preserves_identical_assignments_and_creation_metadata(self):
        self.run_command()
        before = self.metadata_path.read_bytes()
        assignments_before = self.output_path.read_bytes()
        with patch("scripts.split_data.python_version", return_value="3.99.0"):
            self.run_command()
        self.assertEqual(self.metadata_path.read_bytes(), before)
        self.assertEqual(self.output_path.read_bytes(), assignments_before)

    def test_different_metadata_is_rejected_without_overwriting(self):
        self.run_command()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["split"]["random_state"] = 17
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        metadata_before = self.metadata_path.read_bytes()
        assignments_before = self.output_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Existing metadata differ"):
            self.run_command()
        self.assertEqual(self.metadata_path.read_bytes(), metadata_before)
        self.assertEqual(self.output_path.read_bytes(), assignments_before)

    def test_different_valid_split_is_rejected_without_overwriting(self):
        self.run_command()
        existing = pd.read_csv(self.output_path, dtype=str, keep_default_na=False)
        existing["partition"] = existing["partition"].replace(
            {"train": "test", "test": "train"}
        )
        existing.to_csv(self.output_path, index=False)
        before = hashlib.sha256(self.output_path.read_bytes()).hexdigest()

        with self.assertRaisesRegex(ValueError, "differ from newly computed"):
            self.run_command()

        self.assertEqual(
            hashlib.sha256(self.output_path.read_bytes()).hexdigest(), before
        )


class PatientFoldsTest(unittest.TestCase):
    def test_no_patient_overlap(self):
        data = pd.DataFrame(
            {"patient_nbr": ["P1", "P1", "P2", "P3", "P4", "P5"]},
            index=[10, 20, 30, 40, 50, 60],
        )

        folds = list(make_patient_folds(data))
        self.assertEqual(len(folds), 5)

        held_out_positions = []

        for train_positions, validation_positions in folds:
            train_patients = set(data["patient_nbr"].iloc[train_positions])
            validation_patients = set(data["patient_nbr"].iloc[validation_positions])

            self.assertEqual(train_patients & validation_patients, set())
            self.assertEqual(
                set(train_positions) | set(validation_positions),
                set(range(len(data))),
            )

            held_out_positions.extend(validation_positions)

        self.assertEqual(sorted(held_out_positions), list(range(len(data))))
