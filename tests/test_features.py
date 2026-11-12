import unittest

import pandas as pd

from carepath_lab.features import MEDICATION_FEATURES, build_model_inputs


class BuildModelInputsTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "encounter_id": ["E1", "E2", "E3"],
                "patient_nbr": ["P1", "P1", "P2"],
                "time_in_hospital": ["1", "4", "2"],
                "num_medications": ["3", "10", "5"],
                "number_emergency": ["0", "1", "2"],
                "number_inpatient": ["0", "2", "1"],
                "num_lab_procedures": ["12", "34", "56"],
                "num_procedures": ["0", "1", "2"],
                "number_outpatient": ["0", "2", "1"],
                "number_diagnoses": ["3", "7", "5"],
                "age": ["[20-30)", "[20-30)", "[70-80)"],
                "admission_type_id": ["1", "2", "3"],
                "admission_source_id": ["1", "7", "17"],
                "max_glu_serum": ["None", "Norm", ">200"],
                "A1Cresult": ["None", "Norm", ">7"],
                "race": ["Caucasian", "?", "AfricanAmerican"],
                "gender": ["Female", "Unknown/Invalid", "Male"],
                "payer_code": ["MC", "?", "HM"],
                "medical_specialty": ["?", "InternalMedicine", "?"],
                "weight": ["?", "[75-100)", "?"],
                "diag_1": ["250.01", "?", "428"],
                "diag_2": ["V45", "401", "?"],
                "diag_3": ["?", "E999", "250"],
                "readmitted": ["<30", ">30", "NO"],
            },
            index=[10, 30, 70],
        )
        for field in MEDICATION_FEATURES:
            self.data[field] = ["No", "Steady", "Up"]
        self.data["insulin"] = ["Down", "No", "Steady"]
        self.data["change"] = ["No", "Ch", "No"]
        self.data["diabetesMed"] = ["Yes", "Yes", "No"]

    def test_selects_predictors_and_binary_target_with_preserved_alignment(self):
        X, y = build_model_inputs(self.data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(X.shape, (3, 6))
        self.assertEqual(
            list(X.columns),
            [
                "time_in_hospital", "num_medications", "number_emergency",
                "number_inpatient", "age", "admission_type_id",
            ],
        )
        self.assertTrue(X.index.equals(self.data.index))
        self.assertTrue(y.index.equals(X.index))
        self.assertEqual(y.tolist(), [1, 0, 0])
        self.assertEqual(X.loc[30, "time_in_hospital"], 4)
        for column in [
            "time_in_hospital", "num_medications", "number_emergency", "number_inpatient"
        ]:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[column]))
        self.assertEqual(X.loc[70, "age"], "[70-80)")
        self.assertEqual(X.loc[30, "admission_type_id"], "2")
        self.assertFalse({"encounter_id", "patient_nbr", "readmitted"} & set(X.columns))

    def test_does_not_mutate_input_and_returns_independent_predictors(self):
        before = self.data.copy(deep=True)
        X, y = build_model_inputs(self.data)
        pd.testing.assert_frame_equal(self.data, before)
        X.loc[10, "time_in_hospital"] = 99
        y.loc[10] = 0
        pd.testing.assert_frame_equal(self.data, before)

    def test_extended_counts_are_numeric_without_changing_initial_inputs(self):
        before = self.data.copy(deep=True)
        initial_X, initial_y = build_model_inputs(self.data)
        X, y = build_model_inputs(self.data, feature_set="extended")
        self.assertEqual(X.shape, (3, 45))
        self.assertEqual(len(MEDICATION_FEATURES), 23)
        self.assertEqual(len(set(MEDICATION_FEATURES)), 23)
        self.assertEqual(
            X.columns.tolist(),
            [
                "time_in_hospital", "num_medications", "number_emergency",
                "number_inpatient", "num_lab_procedures", "num_procedures",
                "number_outpatient", "number_diagnoses", "age", "admission_type_id",
                "admission_source_id",
                "max_glu_serum", "A1Cresult",
                *MEDICATION_FEATURES, "change", "diabetesMed",
                "race", "gender", "payer_code", "medical_specialty",
                "diag_1", "diag_2", "diag_3",
            ],
        )
        for column in X.columns[:8]:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[column]))
        self.assertEqual(X.loc[30, "num_lab_procedures"], 34)
        self.assertEqual(X.loc[70, "age"], "[70-80)")
        self.assertEqual(X.loc[70, "admission_source_id"], "17")
        self.assertEqual(X.loc[10, "insulin"], "Down")
        self.assertEqual(X.loc[10, "change"], "No")
        self.assertEqual(X.loc[10, "diabetesMed"], "Yes")
        for field in [*MEDICATION_FEATURES, "change", "diabetesMed"]:
            pd.testing.assert_series_equal(X[field], self.data[field])
        for field in ["race", "gender", "payer_code", "medical_specialty"]:
            pd.testing.assert_series_equal(X[field], self.data[field])
            self.assertFalse(X[field].isna().any())
        self.assertEqual(X.loc[30, "race"], "?")
        self.assertEqual(X.loc[30, "gender"], "Unknown/Invalid")
        self.assertNotIn("weight", X.columns)
        self.assertEqual(X.loc[10, "diag_1"], "250.01")
        for field in ["diag_1", "diag_2", "diag_3"]:
            pd.testing.assert_series_equal(X[field], self.data[field])
        for column in ["max_glu_serum", "A1Cresult"]:
            self.assertEqual(X.loc[10, column], "None")
            self.assertEqual(X.loc[30, column], "Norm")
            self.assertFalse(X[column].isna().any())
        self.assertFalse({"encounter_id", "patient_nbr", "readmitted"} & set(X.columns))
        self.assertTrue(X.index.equals(self.data.index))
        pd.testing.assert_series_equal(y, initial_y)
        after_X, after_y = build_model_inputs(self.data)
        pd.testing.assert_frame_equal(after_X, initial_X)
        pd.testing.assert_series_equal(after_y, initial_y)
        pd.testing.assert_frame_equal(self.data, before)

    def test_invalid_extended_numeric_values_are_rejected(self):
        for column in [
            "num_lab_procedures", "num_procedures", "number_outpatient", "number_diagnoses"
        ]:
            with self.subTest(column=column):
                invalid = self.data.copy()
                invalid.loc[30, column] = "abc"
                with self.assertRaises(ValueError):
                    build_model_inputs(invalid, feature_set="extended")

    def test_unknown_target_is_rejected_instead_of_becoming_missing(self):
        for target in ["UNKNOWN", ""]:
            with self.subTest(target=target):
                invalid = self.data.copy()
                invalid.loc[30, "readmitted"] = target
                with self.assertRaisesRegex(ValueError, "Unexpected readmitted categories"):
                    build_model_inputs(invalid)

    def test_invalid_numeric_value_raises_instead_of_becoming_zero_or_missing(self):
        for column in [
            "time_in_hospital", "num_medications", "number_emergency", "number_inpatient"
        ]:
            with self.subTest(column=column):
                invalid = self.data.copy()
                invalid.loc[30, column] = "abc"
                with self.assertRaises(ValueError):
                    build_model_inputs(invalid)
