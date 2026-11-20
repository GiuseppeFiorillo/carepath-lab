import contextlib
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from carepath_lab.data import file_sha256, load_verified_partitions


class VerifiedPartitionsTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "data/raw/diabetic_data.csv"
        self.assignments = self.root / "data/processed/split_assignments.csv"
        self.metadata = self.root / "data/processed/split_metadata.json"
        self.source.parent.mkdir(parents=True)
        self.assignments.parent.mkdir(parents=True)
        data = pd.DataFrame({
            "encounter_id": ["001", "002", "003", "004"],
            "patient_nbr": ["01", "01", "02", "03"],
            "discharge_disposition_id": ["1"] * 4,
            "readmitted": ["<30", "NO", ">30", "NO"],
        })
        data.to_csv(self.source, index=False)
        assignments = data[["encounter_id", "patient_nbr"]].assign(
            partition=["train", "train", "validation", "test"]
        )
        assignments.to_csv(self.assignments, index=False)
        self.metadata.write_text(json.dumps({
            "source": {"sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()}
        }))

    def test_returns_only_train_validation_preserving_ids_and_source_files(self):
        paths = [self.source, self.assignments, self.metadata]
        before = [p.read_bytes() for p in paths]
        with contextlib.chdir(self.root):
            train, validation = load_verified_partitions()
        self.assertEqual(train.encounter_id.tolist(), ["001", "002"])
        self.assertEqual(train.patient_nbr.tolist(), ["01", "01"])
        self.assertEqual(validation.encounter_id.tolist(), ["003"])
        self.assertFalse("004" in set(train.encounter_id) | set(validation.encounter_id))
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_changed_source_is_rejected(self):
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with contextlib.chdir(self.root), self.assertRaisesRegex(ValueError, "SHA-256"):
            load_verified_partitions()

    def test_missing_required_files_are_rejected(self):
        for path in [self.source, self.assignments, self.metadata]:
            content = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(path=path.name), contextlib.chdir(self.root):
                    with self.assertRaises(ValueError):
                        load_verified_partitions()
            finally:
                path.write_bytes(content)

    def test_inconsistent_encounter_patient_pairs_are_rejected(self):
        assignments = pd.read_csv(self.assignments, dtype=str)
        assignments.loc[0, "patient_nbr"] = "WRONG"
        assignments.to_csv(self.assignments, index=False)
        with contextlib.chdir(self.root), self.assertRaises(ValueError):
            load_verified_partitions()

    def test_hash_depends_on_bytes_not_filename(self):
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(file_sha256(self.source), expected)
        renamed = self.source.with_name("renamed.csv")
        self.source.rename(renamed)
        self.assertEqual(file_sha256(renamed), expected)
        renamed.write_bytes(renamed.read_bytes() + b"x")
        self.assertNotEqual(file_sha256(renamed), expected)


if __name__ == "__main__":
    unittest.main()
