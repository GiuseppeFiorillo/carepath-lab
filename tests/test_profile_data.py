"""Check numeric, categorical and duplicate audits using synthetic values only."""

import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

from carepath_lab.cohort import select_analysis_cohort
from scripts.profile_data import main


class NumericProfileTests(unittest.TestCase):
    def test_parse_failures_fractions_and_negative_medications_are_distinct(self):
        data = pd.DataFrame(
            {
                "time_in_hospital": ["1", "invalid", "2.5"],
                "num_lab_procedures": ["1", "invalid", "2.5"],
                "num_procedures": ["0", "invalid", "2.5"],
                "num_medications": ["-2", "invalid", "2.5"],
                "number_diagnoses": ["-1", "2.5", "invalid"],
                "number_outpatient": ["0", "invalid", "1.5"],
                "number_emergency": ["-1", "0", "2"],
                "number_inpatient": ["0", "1", "2"],
                "max_glu_serum": ["None", "Norm", "Norm"],
                "A1Cresult": ["None", "Norm", ">7"],
                "metformin": ["No", "Steady", "steady"],
                "insulin": ["Steady", "Down", "No"],
                "metformin-pioglitazone": "No",
                "change": ["No", "Ch", "Yes"],
                "diabetesMed": ["Yes", "Yes", "Ch"],
                "gender": ["Female", "Male", "Unknown/Invalid"],
                "discharge_disposition_id": ["11", "1", "13"],
                "readmitted": ["NO", "<30", "NO"],
                "diag_1": ["250", "250", "?"],
                "diag_2": ["V45", "?", "401"],
                "diag_3": ["250.13", "250.13", "?"],
            }
        )
        original = data.copy(deep=True)
        output = io.StringIO()
        with (
            patch("scripts.profile_data.pd.read_csv", return_value=data),
            patch("pathlib.Path.mkdir"),
            patch.object(pd.DataFrame, "to_csv"),
            contextlib.redirect_stdout(output),
        ):
            main()

        for message in (
            "Non-numeric stay values: 1",
            "Non-integer stay values: 1",
        ):
            with self.subTest(message=message):
                self.assertIn(message, output.getvalue().splitlines())

        # Read displayed table fields without depending on padding/alignment.
        rows = {
            parts[0]: parts[1:]
            for line in output.getvalue().splitlines()
            if len(parts := [part.strip() for part in line.split("|")]) == 6
        }
        expected = {
            "lab procedures": (1.0, 2.5, 1, 1, 0),
            "procedures": (0.0, 2.5, 1, 1, 0),
            "medications": (-2.0, 2.5, 1, 1, 1),
            "diagnoses": (-1.0, 2.5, 1, 1, 1),
            "prior outpatient visits": (0.0, 1.5, 1, 1, 0),
            "prior emergency visits": (-1.0, 2.0, 0, 0, 1),
            "prior inpatient encounters": (0.0, 2.0, 0, 0, 0),
        }
        for label, values in expected.items():
            with self.subTest(label=label):
                actual = tuple(float(value) for value in rows[label])
                self.assertEqual(actual, values)
        self.assertIn(
            "Original dataset: 3 encounters | <30: 1 (33.33%) | Excluded: 0",
            output.getvalue(),
        )
        self.assertIn(
            "Working cohort: exclude 11/19/20/21: 2 encounters | <30: 1 (50.00%) | Excluded: 1",
            output.getvalue(),
        )
        medication_section = output.getvalue().split("9b. Medication category check", 1)[1]
        self.assertIn("Unexpected categories: ['steady']", medication_section)
        self.assertIn("metformin-pioglitazone is a constant column", medication_section)
        indicator_section = output.getvalue().split("9c. Medication indicator checks", 1)[1]
        self.assertIn("Unexpected categories: ['Yes']", indicator_section)
        self.assertIn("Unexpected categories: ['Ch']", indicator_section)
        diagnosis_section = output.getvalue().split("10. Diagnosis codes", 1)[1].split("Export", 1)[0]
        cardinality_lines = [
            line for line in diagnosis_section.splitlines()
            if line.startswith("Number of distinct codes:")
        ]
        self.assertEqual(cardinality_lines, [
            "Number of distinct codes: 1",
            "Number of distinct codes: 2",
            "Number of distinct codes: 1",
        ])
        self.assertIn("V45", diagnosis_section)
        self.assertIn("250.13", diagnosis_section)
        self.assertNotIn("?", diagnosis_section)
        pd.testing.assert_frame_equal(data, original)


class AnalysisCohortTests(unittest.TestCase):
    def test_excludes_all_selected_death_codes_but_retains_hospice_and_transfer(self):
        source = pd.DataFrame(
            {"discharge_disposition_id": ["11", "19", "20", "21", "13", "14", "2", "1"]}
        )
        original = source.copy(deep=True)
        selected = select_analysis_cohort(source)
        self.assertEqual(selected["discharge_disposition_id"].tolist(), ["13", "14", "2", "1"])
        pd.testing.assert_frame_equal(source, original)
        selected.loc[selected.index[0], "discharge_disposition_id"] = "changed"
        pd.testing.assert_frame_equal(source, original)

    def test_selection_can_be_empty(self):
        source = pd.DataFrame({"discharge_disposition_id": ["11", "19", "20", "21"]})
        self.assertTrue(select_analysis_cohort(source).empty)


class DuplicateAuditTests(unittest.TestCase):
    def test_counts_all_matches_without_merging_patients_or_distinct_values(self):
        # Rows 0/1 match except for encounter_id. The other rows differ in
        # patient, age or target and must not enter that duplicate group.
        data = pd.DataFrame({
            "encounter_id": ["E1", "E2", "E3", "E4", "E5"],
            "patient_nbr": ["P1", "P1", "P2", "P1", "P1"],
            "age": ["[50-60)", "[50-60)", "[50-60)", "[60-70)", "[50-60)"],
            "readmitted": ["NO", "NO", "NO", "NO", "<30"],
            "diag_1": "250",
            "diag_2": "401",
            "diag_3": "V45",
            "time_in_hospital": "1",
            "num_lab_procedures": "1",
            "num_procedures": "0",
            "num_medications": "1",
            "number_diagnoses": "1",
            "number_outpatient": "0",
            "number_emergency": "0",
            "number_inpatient": "0",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "metformin": "No",
            "insulin": "Steady",
            "metformin-pioglitazone": "No",
            "change": "No",
            "diabetesMed": "Yes",
            "gender": "Female",
            "discharge_disposition_id": "1",
        })
        original = data.copy(deep=True)
        output = io.StringIO()
        with (
            patch("scripts.profile_data.pd.read_csv", return_value=data),
            patch("pathlib.Path.mkdir"),
            patch.object(pd.DataFrame, "to_csv"),
            contextlib.redirect_stdout(output),
        ):
            main()
        self.assertIn("Duplicate rows (all occurrences): 2", output.getvalue().splitlines())
        pd.testing.assert_frame_equal(data, original)
