"""Check section parsing using synthetic mapping CSVs only."""

import contextlib
import io
import unittest
from unittest.mock import mock_open, patch

import pandas as pd

from scripts import audit_administrative_mappings


class AdministrativeMappingTests(unittest.TestCase):
    def run_audit(self, fixture, observed=None):
        output = io.StringIO()
        if observed is None:
            observed = pd.DataFrame({
                name: ["1"] for name in (
                    "admission_type_id", "discharge_disposition_id", "admission_source_id"
                )
            })
        with (
            patch("builtins.open", mock_open(read_data=fixture)),
            patch.object(audit_administrative_mappings.pd, "read_csv", return_value=observed),
            contextlib.redirect_stdout(output),
        ):
            audit_administrative_mappings.main()
        return output.getvalue()

    def test_reused_codes_and_literal_null_descriptions_are_accepted(self):
        fixture = (
            "admission_type_id,description\n1,Emergency\n,\n"
            "discharge_disposition_id,description\n1,NULL\n,\n"
            "admission_source_id,description\n1, Physician Referral\n"
        )
        output = self.run_audit(fixture)
        for name in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
            self.assertIn(f"{name} has ", output)
        self.assertEqual(output.count("Unmapped codes: []"), 3)

    def test_observed_unmapped_code_is_reported(self):
        fixture = (
            "admission_type_id,description\n1,Emergency\n,\n"
            "discharge_disposition_id,description\n1,NULL\n,\n"
            "admission_source_id,description\n1, Physician Referral\n"
        )
        observed = pd.DataFrame({
            "admission_type_id": ["99"],
            "discharge_disposition_id": ["1"],
            "admission_source_id": ["1"],
        })
        output = self.run_audit(fixture, observed)
        self.assertIn("Unmapped codes: ['99']", output)
        self.assertEqual(output.count("Unmapped codes: []"), 2)

    def test_unused_mapping_code_is_not_a_coverage_failure(self):
        fixture = (
            "admission_type_id,description\n1,Emergency\n2,Urgent\n,\n"
            "discharge_disposition_id,description\n1,NULL\n,\n"
            "admission_source_id,description\n1, Physician Referral\n"
        )
        output = self.run_audit(fixture)
        self.assertEqual(output.count("Unmapped codes: []"), 3)

    def test_invalid_mapping_records_raise(self):
        prefix = "admission_type_id,description\n"
        cases = [
            (prefix + "1,Emergency,extra\n", "Unexpected number of fields"),
            (prefix + "1,Emergency\n1,Urgent\n", "Duplicate code"),
            ("1,Emergency\n", "Current section is None"),
            (prefix + ",Emergency\n", "Code is empty"),
            (prefix + "1,   \n", "Description is empty"),
            (prefix + "1,Emergency\n,\n2,Urgent\n", "Current section is None"),
            ("admission_type_id,wrong_header\n", "Unexpected mapping section header"),
        ]
        for fixture, message in cases:
            with self.subTest(fixture=fixture):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_audit(fixture)

    def test_unavailable_labels_count_encounters_not_distinct_codes(self):
        fixture = (
            "admission_type_id,description\n1,Emergency\n"
            "5,Not Available\n6,NULL\n8,Not Mapped\n,\n"
            "discharge_disposition_id,description\n1,Discharged to home\n"
            "18,NULL\n25,Not Mapped\n26,Unknown/Invalid\n,\n"
            "admission_source_id,description\n1, Physician Referral\n"
            "9, Not Available\n20, Not Mapped\n"
        )
        observed = pd.DataFrame({
            "admission_type_id": ["5", "5", "6", "8", "1", "1"],
            "discharge_disposition_id": ["18", "25", "26", "1", "1", "1"],
            "admission_source_id": ["9", "20", "1", "1", "1", "1"],
        })
        original = observed.copy(deep=True)
        output = self.run_audit(fixture, observed)
        summaries = [
            line for line in output.splitlines()
            if line.startswith("Rows with unavailable codes:")
        ]
        self.assertEqual(summaries, [
            "Rows with unavailable codes: 4 (66.67%)",
            "Rows with unavailable codes: 3 (50.00%)",
            "Rows with unavailable codes: 2 (33.33%)",
        ])
        pd.testing.assert_frame_equal(observed, original)

    def test_malformed_header_inside_active_section_is_not_treated_as_data(self):
        fixture = (
            "admission_type_id,description\n1,Emergency\n"
            "admission_source_id,wrong_header\n"
        )
        with self.assertRaisesRegex(ValueError, "Unexpected mapping section header"):
            self.run_audit(fixture)
