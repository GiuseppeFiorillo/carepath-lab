"""Check the shared learner-written CSV validation contract."""

import csv
import io
import unittest

from carepath_lab.validation import read_header, validate_and_extract_record


class HeaderTests(unittest.TestCase):
    def test_reads_required_columns_in_source_order(self):
        reader = csv.reader(io.StringIO("readmitted,note,patient_nbr,encounter_id\n"))
        self.assertEqual(read_header(reader), ["readmitted", "note", "patient_nbr", "encounter_id"])

    def test_rejects_empty_file_and_blank_header(self):
        for content in ("", "\n"):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "no header"):
                read_header(csv.reader(io.StringIO(content)))

    def test_rejects_duplicate_or_blank_column_names(self):
        headers = [
            "encounter_id,patient_nbr,readmitted,readmitted\n",
            "encounter_id,patient_nbr,readmitted,\n",
            "encounter_id,patient_nbr,readmitted,   \n",
        ]
        for header in headers:
            with self.subTest(header=header), self.assertRaises(ValueError):
                read_header(csv.reader(io.StringIO(header)))

    def test_rejects_missing_required_fields(self):
        for header in ("encounter_id,patient_nbr\n", "readmitted,patient_nbr\n", "encounter_id,readmitted\n"):
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, "missing required"):
                read_header(csv.reader(io.StringIO(header)))


class ValidateRecordTest(unittest.TestCase):
    def test_valid_record_returns_selected_values(self):
        columns = ["encounter_id", "patient_nbr", "readmitted"]
        row = ["encounter-A", "patient-A", "<30"]
        allowed_values = {"<30", ">30", "NO"}

        result = validate_and_extract_record(
            row,
            columns,
            columns.index("encounter_id"),
            columns.index("patient_nbr"),
            columns.index("readmitted"),
            allowed_values,
        )

        self.assertEqual(result, ("encounter-A", "patient-A", "<30"))

    def test_record_with_missing_field_is_rejected(self):
        columns = ["encounter_id", "patient_nbr", "readmitted"]
        row = ["encounter-B", "patient-B"]
        allowed_values = {"<30", ">30", "NO"}

        with self.assertRaises(ValueError):
            validate_and_extract_record(
                row,
                columns,
                columns.index("encounter_id"),
                columns.index("patient_nbr"),
                columns.index("readmitted"),
                allowed_values,
            )

    def test_record_with_extra_field_is_rejected(self):
        columns = ["encounter_id", "patient_nbr", "readmitted"]
        with self.assertRaisesRegex(ValueError, "Unexpected number of fields"):
            validate_and_extract_record(["E1", "P1", "NO", "extra"], columns, 0, 1, 2, {"NO"})

    def test_empty_identifiers_and_target_are_rejected(self):
        columns = ["encounter_id", "patient_nbr", "readmitted"]
        for row in ([" ", "P1", "NO"], ["E1", "\t", "NO"], ["E1", "P1", ""]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_and_extract_record(row, columns, 0, 1, 2, {"<30", ">30", "NO"})

    def test_record_with_unknown_target(self):
        columns = ["encounter_id", "patient_nbr", "readmitted"]
        row = ["encounter-C", "patient-C", "UNKNOWN"]
        allowed_values = {"<30", ">30", "NO"}

        with self.assertRaisesRegex(ValueError, "Unexpected readmitted category"):
            validate_and_extract_record(
                row,
                columns,
                columns.index("encounter_id"),
                columns.index("patient_nbr"),
                columns.index("readmitted"),
                allowed_values,
            )
