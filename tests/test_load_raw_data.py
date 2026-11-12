"""Exercise loader orchestration with synthetic CSV data and a mocked connection."""

import contextlib
import csv
import io
import unittest
from unittest.mock import MagicMock, mock_open, patch

import psycopg
from psycopg import sql

from scripts import load_raw_data


class RawLoaderTests(unittest.TestCase):
    def setUp(self):
        # Include names needing quoting and a literal nonmeasurement marker.
        self.columns = [
            "encounter_id", "patient_nbr", "readmitted",
            "A1Cresult", "glyburide-metformin",
        ]
        self.row = ["encounter-A", "patient-A", "<30", "None", "No"]
        self.connection = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.fetchall.return_value = [(name,) for name in self.columns]
        self.cursor.fetchone.return_value = (0,)
        self.output = io.StringIO()

    def run_loader(self, rows, header=None):
        fixture = io.StringIO(newline="")
        writer = csv.writer(fixture)
        writer.writerow(self.columns if header is None else header)
        writer.writerows(rows)
        with (
            patch.object(load_raw_data.psycopg, "connect") as connect,
            patch("builtins.open", mock_open(read_data=fixture.getvalue())),
            contextlib.redirect_stdout(self.output),
        ):
            connect.return_value = self.connection
            load_raw_data.main()

    def insert_calls(self):
        return [
            call for call in self.cursor.execute.call_args_list
            if isinstance(call.args[0], sql.Composed)
        ]

    def test_valid_rows_preserve_values_order_and_quoted_names(self):
        rows = [self.row, ["encounter-B", "patient-A", "NO", "Norm", "No"]]
        self.run_loader(rows)
        calls = self.insert_calls()
        self.assertEqual([call.args[1] for call in calls], rows)
        self.assertEqual(
            calls[0].args[0].as_string(),
            'INSERT INTO encounters_raw ("encounter_id", "patient_nbr", '
            '"readmitted", "A1Cresult", "glyburide-metformin") '
            'VALUES (%s, %s, %s, %s, %s);',
        )
        self.connection.__exit__.assert_called_once_with(None, None, None)
        self.assertIn("Loaded encounters: 2", self.output.getvalue())

    def test_header_only_raises_before_connection_exit(self):
        with self.assertRaisesRegex(ValueError, "CSV contains no encounter records"):
            self.run_loader([])
        self.assert_failure_reaches_connection(ValueError)
        self.assertEqual(self.insert_calls(), [])

    def test_invalid_second_record_propagates_after_first_insert(self):
        invalid = ["encounter-B", "patient-A", "OTHER", "None", "No"]
        with self.assertRaisesRegex(ValueError, "Unexpected readmitted category"):
            self.run_loader([self.row, invalid])
        self.assertEqual(len(self.insert_calls()), 1)
        self.assert_failure_reaches_connection(ValueError)

    def test_nonempty_destination_is_rejected_without_insert(self):
        self.cursor.fetchone.return_value = (1,)
        with self.assertRaisesRegex(ValueError, "Raw table must be empty"):
            self.run_loader([self.row])
        self.assertEqual(self.insert_calls(), [])
        self.assert_failure_reaches_connection(ValueError)

    def test_reordered_header_is_rejected_without_insert(self):
        header = self.columns.copy()
        header[0], header[1] = header[1], header[0]
        with self.assertRaisesRegex(ValueError, "CSV header does not match"):
            self.run_loader([self.row], header=header)
        self.assertEqual(self.insert_calls(), [])
        self.assert_failure_reaches_connection(ValueError)

    def test_insert_error_propagates_without_success_message(self):
        def execute(query, parameters=None):
            if isinstance(query, sql.Composed):
                raise psycopg.errors.CheckViolation("Synthetic constraint failure")

        self.cursor.execute.side_effect = execute
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.run_loader([self.row])
        self.assert_failure_reaches_connection(psycopg.errors.CheckViolation)

    def test_connection_exit_error_prevents_success_message(self):
        self.connection.__exit__.side_effect = psycopg.OperationalError(
            "Synthetic commit failure"
        )
        with self.assertRaises(psycopg.OperationalError):
            self.run_loader([self.row])
        self.assertNotIn("Loaded encounters:", self.output.getvalue())

    def assert_failure_reaches_connection(self, exception_type):
        self.assertIs(self.connection.__exit__.call_args.args[0], exception_type)
        self.assertNotIn("Loaded encounters:", self.output.getvalue())
