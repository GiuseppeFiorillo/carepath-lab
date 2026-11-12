"""Check exact source/import reconciliation using synthetic rows only."""

import contextlib
import csv
import io
import unittest
from unittest.mock import MagicMock, mock_open, patch

from psycopg import sql

from scripts import verify_raw_import


class VerifyRawImportTests(unittest.TestCase):
    def setUp(self):
        self.columns = [
            "encounter_id", "patient_nbr", "readmitted", "age", "A1Cresult"
        ]
        self.rows = [
            ("E1", "P1", "<30", "[50-60)", "None"),
            ("E2", "P1", "NO", "[50-60)", "Norm"),
        ]
        self.output = io.StringIO()

    def run_verifier(self, database_rows, source_rows=None, database_columns=None):
        fixture = io.StringIO(newline="")
        writer = csv.writer(fixture)
        writer.writerow(self.columns)
        writer.writerows(self.rows if source_rows is None else source_rows)
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        names = self.columns if database_columns is None else database_columns
        cursor.fetchall.return_value = [(name,) for name in names]
        cursor.__iter__.side_effect = lambda: iter(database_rows)
        with (
            patch.object(verify_raw_import.psycopg, "connect") as connect,
            patch("builtins.open", mock_open(read_data=fixture.getvalue())),
            contextlib.redirect_stdout(self.output),
        ):
            self.connect = connect
            connect.return_value.__enter__.return_value = connection
            try:
                verify_raw_import.main()
            finally:
                # The verification must issue only SELECT statements.
                for call in cursor.execute.call_args_list:
                    query = call.args[0]
                    text = query.as_string() if isinstance(query, sql.Composed) else query
                    self.assertTrue(text.lstrip().startswith("SELECT"))

    def assert_discrepancy(self, database_rows, expected_counts):
        with self.assertRaisesRegex(ValueError, "Database records do not match"):
            self.run_verifier(database_rows)
        for label, count in expected_counts.items():
            self.assertIn(f"{label}: {count}", self.output.getvalue().splitlines())
        self.assertNotIn("Raw import verified:", self.output.getvalue())

    def test_identical_records_in_different_order_pass(self):
        self.run_verifier(list(reversed(self.rows)))
        self.assertIn("Database encounters: 2", self.output.getvalue())
        for label in ("Missing records", "Unexpected records", "Mismatched records"):
            self.assertIn(f"{label}: 0", self.output.getvalue().splitlines())
        self.assertIn("Raw import verified:", self.output.getvalue())

    def test_missing_record_fails(self):
        self.assert_discrepancy(self.rows[:1], {
            "Missing records": 1, "Unexpected records": 0, "Mismatched records": 0,
        })

    def test_unexpected_record_fails(self):
        extra = ("E3", "P2", "NO", "[20-30)", "None")
        self.assert_discrepancy(self.rows + [extra], {
            "Missing records": 0, "Unexpected records": 1, "Mismatched records": 0,
        })

    def test_changed_age_fails_despite_matching_ids_and_targets(self):
        changed = ("E2", "P1", "NO", "[60-70)", "Norm")
        self.assert_discrepancy([self.rows[0], changed], {
            "Missing records": 0, "Unexpected records": 0, "Mismatched records": 1,
        })

    def test_missing_and_extra_records_fail_despite_matching_totals(self):
        extra = ("E3", "P2", "NO", "[50-60)", "Norm")
        self.assert_discrepancy([self.rows[0], extra], {
            "Database encounters": 2, "Missing records": 1,
            "Unexpected records": 1, "Mismatched records": 0,
        })

    def test_sql_null_does_not_match_literal_none_string(self):
        changed = ("E1", "P1", "<30", "[50-60)", None)
        self.assert_discrepancy([changed, self.rows[1]], {
            "Missing records": 0, "Unexpected records": 0, "Mismatched records": 1,
        })

    def test_duplicate_source_id_fails_before_connecting(self):
        with self.assertRaisesRegex(ValueError, "Duplicate encounter ID"):
            self.run_verifier(self.rows, source_rows=[self.rows[0], self.rows[0]])
        self.connect.assert_not_called()

    def test_reordered_schema_fails_without_success_message(self):
        with self.assertRaisesRegex(ValueError, "CSV header does not match"):
            self.run_verifier(self.rows, database_columns=list(reversed(self.columns)))
        self.assertNotIn("Raw import verified:", self.output.getvalue())
