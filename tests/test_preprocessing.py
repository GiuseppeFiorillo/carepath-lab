import unittest

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from carepath_lab.features import FEATURE_SETS, MEDICATION_FEATURES, build_preprocessor


class PreprocessingTest(unittest.TestCase):
    def test_native_indicator_marks_original_missingness_not_mode_membership(self):
        train = self.train.assign(admission_type_id=["1", "?"])
        validation = pd.concat([self.validation] * 3, ignore_index=True).assign(
            admission_type_id=["?", "1", "3"]
        )
        original = validation.copy(deep=True)
        preprocessor = build_preprocessor(
            native_categorical=True, scale_numeric=False,
            imputed_categorical_features=("admission_type_id",),
            add_missing_indicators=True,
        )
        preprocessor.fit(train)
        transformed = preprocessor.transform(validation)
        self.assertEqual(transformed.shape, (3, 7))
        self.assertEqual(transformed["admission_type_id"].tolist(), ["1", "1", "3"])
        self.assertEqual(
            transformed["missingindicator_admission_type_id"].tolist(),
            [True, False, False],
        )
        imputer = preprocessor.named_transformers_["categorical_imputed"].named_steps["imputer"]
        np.testing.assert_array_equal(imputer.statistics_, ["1"])
        pd.testing.assert_frame_equal(validation, original)

    def test_indicator_columns_depend_only_on_missingness_seen_during_fit(self):
        preprocessor = build_preprocessor(
            native_categorical=True, scale_numeric=False,
            imputed_categorical_features=("admission_type_id",),
            add_missing_indicators=True,
        )
        preprocessor.fit(self.train)
        validation = self.validation.assign(admission_type_id=["?"])
        transformed = preprocessor.transform(validation)
        self.assertEqual(set(transformed.columns), set(self.train.columns))
        self.assertEqual(transformed["admission_type_id"].tolist(), ["1"])

    def test_native_output_preserves_declared_columns_and_drops_identifiers(self):
        train = self.train.assign(encounter_id=["E1", "E2"], readmitted=["NO", "<30"])
        original = train.copy(deep=True)
        preprocessor = build_preprocessor(native_categorical=True, scale_numeric=False)
        transformed = preprocessor.fit_transform(train)
        pd.testing.assert_frame_equal(transformed, self.train)
        pd.testing.assert_frame_equal(preprocessor.transform(self.validation), self.validation)
        pd.testing.assert_frame_equal(train, original)

    def test_native_imputation_preserves_values_names_index_and_training_statistics(self):
        train = self.train.assign(admission_type_id=["1", "?"])
        validation = pd.concat([self.validation] * 3, ignore_index=True).assign(
            admission_type_id=["?", "3", "None"]
        )
        validation.index = [90, 10, 70]
        original_train = train.copy(deep=True)
        original_validation = validation.copy(deep=True)
        preprocessor = build_preprocessor(
            native_categorical=True, scale_numeric=False,
            imputed_categorical_features=("admission_type_id",),
        )
        transformed_train = preprocessor.fit_transform(train)
        transformed_validation = preprocessor.transform(validation)
        self.assertIsInstance(transformed_validation, pd.DataFrame)
        self.assertEqual(list(transformed_validation.columns), list(train.columns))
        self.assertEqual(transformed_validation["admission_type_id"].tolist(), ["1", "3", "None"])
        self.assertEqual(transformed_train["admission_type_id"].tolist(), ["1", "1"])
        untouched = [name for name in train.columns if name != "admission_type_id"]
        pd.testing.assert_frame_equal(transformed_validation[untouched], validation[untouched])
        imputer = preprocessor.named_transformers_["categorical_imputed"].named_steps["imputer"]
        np.testing.assert_array_equal(imputer.statistics_, ["1"])
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(validation, original_validation)

    def test_extended_native_output_imputes_specialty_only(self):
        config = FEATURE_SETS["extended"]
        train = pd.DataFrame({
            **{name: [1, 2] for name in config["numeric"]},
            **{name: ["None", "?"] for name in config["categorical"]},
            "medical_specialty": ["InternalMedicine", "?"],
        }, index=[10, 30])
        preprocessor = build_preprocessor(
            feature_set="extended", native_categorical=True, scale_numeric=False,
            imputed_categorical_features=("medical_specialty",),
        )
        transformed = preprocessor.fit_transform(train)
        self.assertEqual(transformed.shape, (2, 45))
        self.assertEqual(set(transformed.columns), set(train.columns))
        self.assertEqual(transformed["medical_specialty"].tolist(), ["InternalMedicine"] * 2)
        untouched = [name for name in train.columns if name != "medical_specialty"]
        pd.testing.assert_frame_equal(transformed[untouched], train[untouched])

    def setUp(self):
        self.train = pd.DataFrame(
            {
                "time_in_hospital": [1, 3],
                "num_medications": [2, 6],
                "number_emergency": [0, 2],
                "number_inpatient": [1, 5],
                "age": ["[20-30)", "[70-80)"],
                "admission_type_id": ["1", "3"],
            }
        )
        self.validation = pd.DataFrame(
            {
                "time_in_hospital": [5],
                "num_medications": [10],
                "number_emergency": [4],
                "number_inpatient": [9],
                "age": ["[20-30)"],
                "admission_type_id": ["2"],
            }
        )

    def test_builder_returns_an_unfitted_object(self):
        preprocessor = build_preprocessor()
        with self.assertRaises(NotFittedError):
            preprocessor.transform(self.train)

    def test_validation_uses_training_statistics_and_fixed_category_vocabulary(self):
        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(self.train)
        scaler = preprocessor.named_transformers_["numeric"]
        encoder = preprocessor.named_transformers_["categorical"]
        means = scaler.mean_.copy()
        scales = scaler.scale_.copy()
        categories = [values.copy() for values in encoder.categories_]

        transformed_validation = preprocessor.transform(self.validation)

        self.assertEqual(transformed_train.shape, (2, 8))
        np.testing.assert_allclose(transformed_train[:, :4], [[-1] * 4, [1] * 4])
        np.testing.assert_allclose(
            transformed_validation, [[3, 3, 3, 3, 1, 0, 0, 0]]
        )
        np.testing.assert_array_equal(scaler.mean_, [2, 4, 1, 3])
        np.testing.assert_array_equal(scaler.mean_, means)
        np.testing.assert_array_equal(scaler.scale_, scales)
        for before, after in zip(categories, encoder.categories_):
            np.testing.assert_array_equal(before, after)

    def test_undeclared_columns_are_dropped_and_inputs_are_unchanged(self):
        extra = self.train.assign(encounter_id=["E1", "E2"], readmitted=["NO", "<30"])
        before = extra.copy(deep=True)
        expected = build_preprocessor().fit_transform(self.train)
        actual = build_preprocessor().fit_transform(extra)
        np.testing.assert_allclose(actual, expected)
        pd.testing.assert_frame_equal(extra, before)

    def test_extended_counts_are_scaled_with_training_only_statistics(self):
        train = self.train.assign(
            num_lab_procedures=[10, 30],
            num_procedures=[0, 2],
            number_outpatient=[0, 4],
            number_diagnoses=[2, 6],
            admission_source_id=["1", "7"],
            max_glu_serum=["None", "Norm"],
            A1Cresult=["Norm", "None"],
            race=["Caucasian", "?"],
            gender=["Female", "Male"],
            payer_code=["MC", "?"],
            medical_specialty=["?", "InternalMedicine"],
            diag_1=["250", "?"],
            diag_2=["V45", "250.01"],
            diag_3=["?", "E999"],
        )
        validation = self.validation.assign(
            num_lab_procedures=[50],
            num_procedures=[4],
            number_outpatient=[8],
            number_diagnoses=[10],
            admission_source_id=["17"],
            max_glu_serum=["Norm"],
            A1Cresult=["None"],
            race=["?"],
            gender=["Unknown/Invalid"],
            payer_code=["?"],
            medical_specialty=["?"],
            diag_1=["401"],
            diag_2=["V45"],
            diag_3=["?"],
        )
        for field in MEDICATION_FEATURES:
            train[field] = ["No", "No"]
            validation[field] = ["No"]
        train["metformin"] = ["No", "Steady"]
        validation["metformin"] = ["Up"]
        train["change"] = ["No", "Ch"]
        validation["change"] = ["No"]
        train["diabetesMed"] = ["Yes", "No"]
        validation["diabetesMed"] = ["Yes"]
        preprocessor = build_preprocessor(feature_set="extended")
        transformed_train = preprocessor.fit_transform(train)
        transformed_validation = preprocessor.transform(validation)
        self.assertTrue(issparse(transformed_train))
        self.assertTrue(issparse(transformed_validation))
        # Densify only this tiny fixture to check unchanged numerical encoding.
        transformed_train = transformed_train.toarray()
        transformed_validation = transformed_validation.toarray()
        scaler = preprocessor.named_transformers_["numeric"]

        self.assertEqual(transformed_train.shape, (2, 60))
        self.assertEqual(transformed_validation.shape, (1, 60))
        self.assertEqual(scaler.n_features_in_, 8)
        np.testing.assert_allclose(scaler.mean_, [2, 4, 1, 3, 20, 1, 2, 4])
        np.testing.assert_allclose(transformed_train[:, :8], [[-1] * 8, [1] * 8])
        np.testing.assert_allclose(
            transformed_validation,
            [[3] * 8 + [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]
             + [0, 0] + [1] * 22 + [0, 1, 0, 1]
             + [1, 0, 0, 0, 1, 0, 1, 0] + [0, 0, 0, 1, 1, 0]],
        )
        # Unseen source code: retain the encounter, zero only its one-hot block.
        np.testing.assert_array_equal(transformed_validation[:, 12:14], [[0, 0]])
        # Nonmeasurement and normal results receive distinct indicators per test.
        encoder = preprocessor.named_transformers_["categorical"]
        for categories in encoder.categories_[3:5]:
            np.testing.assert_array_equal(categories, ["None", "Norm"])
        np.testing.assert_array_equal(transformed_validation[:, 14:18], [[0, 1, 1, 0]])
        # Unseen dosage status zeros the drug block, not the encounter/other drugs.
        np.testing.assert_array_equal(transformed_validation[:, 18:20], [[0, 0]])
        np.testing.assert_array_equal(encoder.categories_[28], ["Ch", "No"])
        np.testing.assert_array_equal(encoder.categories_[29], ["No", "Yes"])
        # Literal missing markers get learned indicators; unseen labels stay zero.
        for offset in [30, 32, 33]:
            self.assertIn("?", encoder.categories_[offset])
        np.testing.assert_array_equal(
            transformed_validation[:, 46:54], [[1, 0, 0, 0, 1, 0, 1, 0]]
        )
        np.testing.assert_array_equal(
            transformed_validation[:, -6:], [[0, 0, 0, 1, 1, 0]]
        )
        dense_reference = clone(preprocessor).set_params(
            categorical__sparse_output=False, sparse_threshold=0.0
        )
        np.testing.assert_allclose(
            dense_reference.fit_transform(train), transformed_train
        )
        np.testing.assert_allclose(
            dense_reference.transform(validation), transformed_validation
        )
        self.assertIn("categorical__examide_No", preprocessor.get_feature_names_out())
        # Building/using an extended configuration must not change the default.
        initial = build_preprocessor().fit_transform(train)
        self.assertEqual(initial.shape, (2, 8))
        self.assertFalse(issparse(initial))

    def test_imputation_uses_training_statistics(self):
        train = self.train.assign(admission_type_id=["1", "?"])
        validation = pd.concat(
            [self.validation] * 3,
            ignore_index=True,
        ).assign(admission_type_id=["?", "2", "2"])

        preprocessor = build_preprocessor(
            imputed_categorical_features=("admission_type_id",)
        )
        preprocessor.fit(train)

        imputer = preprocessor.named_transformers_[
            "categorical_imputed"
        ].named_steps["imputer"]

        # Check the learned replacement
        np.testing.assert_array_equal(imputer.statistics_, ["1"])

        transformed_validation = preprocessor.transform(validation)

        # Check that validation did not change the learned replacement.
        np.testing.assert_array_equal(imputer.statistics_, ["1"])

        imputed_values = imputer.transform(
            validation[["admission_type_id"]]
        )
        np.testing.assert_array_equal(imputed_values, [["1"], ["2"], ["2"]])

        np.testing.assert_array_equal(
            transformed_validation[:, -1],
            [1, 0, 0]
        )

    def test_imputation_rejects_non_categorical_features(self):
        with self.assertRaisesRegex(ValueError, "Invalid imputation features"):
            build_preprocessor(
                imputed_categorical_features=("time_in_hospital",),
            )
    
    def test_imputation_rejects_duplicate_features(self):
        with self.assertRaisesRegex(ValueError, "Duplicate imputation features"):
            build_preprocessor(
                imputed_categorical_features=("admission_type_id", "admission_type_id",),
            )
