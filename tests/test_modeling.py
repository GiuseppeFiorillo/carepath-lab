import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from carepath_lab.features import FEATURE_SETS, MEDICATION_FEATURES, build_model_inputs
from carepath_lab.modeling import (
    build_baseline_model,
    build_catboost_model,
    build_lightgbm_model,
    build_random_forest_model,
    evaluate_model_cv,
    fit_and_predict_validation,
)
from carepath_lab.splitting import make_patient_folds


class ModelingFixture(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "time_in_hospital": [1, 2, 3, 4],
                "num_medications": [2, 4, 6, 8],
                "number_emergency": [0, 0, 1, 1],
                "number_inpatient": [0, 1, 0, 1],
                "age": ["[20-30)", "[70-80)", "[20-30)", "[70-80)"],
                "admission_type_id": ["1", "3", "1", "3"],
            },
            index=[10, 30, 50, 70],
        )
        self.target = pd.Series([0, 1, 0, 1], index=self.train.index)
        self.validation = self.train.iloc[[0]].copy()
        self.validation["time_in_hospital"] = 14
        self.validation["admission_type_id"] = "2"


class CatBoostModelTest(ModelingFixture):
    def test_reviewed_native_imputation_uses_only_reviewed_predictors(self):
        self.check_native_imputation(False, feature_set="reviewed")

    def test_optional_imputation_reaches_native_classifier_without_scaling(self):
        for add_indicators in [False, True]:
            with self.subTest(add_missing_indicators=add_indicators):
                self.check_native_imputation(add_indicators)

    def check_native_imputation(self, add_indicators, feature_set="extended"):
        config = FEATURE_SETS[feature_set]
        train = pd.DataFrame({
            **{name: [1, 2, 3, 4] for name in config["numeric"]},
            **{name: ["None", "recorded"] * 2 for name in config["categorical"]},
            "medical_specialty": ["InternalMedicine", "?", "InternalMedicine", "Cardiology"],
        }, index=[10, 30, 50, 70])
        validation = train.iloc[:2].copy().assign(
            medical_specialty=["?", "UnseenSpecialty"]
        )
        original_train = train.copy(deep=True)
        original_validation = validation.copy(deep=True)
        model = build_catboost_model(
            feature_set=feature_set, n_jobs=1,
            imputed_categorical_features=("medical_specialty",),
            add_missing_indicators=add_indicators,
        )
        self.assertEqual(list(model.named_steps), ["preprocessor", "classifier"])
        model.set_params(classifier__iterations=3, classifier__depth=2)
        classifier = model.named_steps["classifier"]
        classifier.fit = Mock(wraps=classifier.fit)
        model.fit(train, self.target)
        fit_inputs = classifier.fit.call_args.args[0]
        self.assertIsInstance(fit_inputs, pd.DataFrame)
        expected_names = set(train.columns)
        indicator_name = "missingindicator_medical_specialty"
        if add_indicators:
            expected_names.add(indicator_name)
            self.assertEqual(fit_inputs[indicator_name].tolist(), [False, True, False, False])
            self.assertNotIn(
                classifier.feature_names_.index(indicator_name),
                classifier.get_cat_feature_indices(),
            )
        self.assertEqual(set(classifier.feature_names_), expected_names)
        expected_count = 41 if feature_set == "reviewed" else 45
        self.assertEqual(len(classifier.feature_names_), expected_count + int(add_indicators))
        self.assertEqual(fit_inputs["medical_specialty"].tolist(), [
            "InternalMedicine", "InternalMedicine", "InternalMedicine", "Cardiology",
        ])
        untouched = [name for name in train.columns if name != "medical_specialty"]
        pd.testing.assert_frame_equal(fit_inputs[untouched], train[untouched])
        probabilities = model.predict_proba(validation)
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])
        classifier.fit.assert_called_once()
        preprocessor = model.named_steps["preprocessor"]
        imputer = preprocessor.named_transformers_["categorical_imputed"].named_steps["imputer"]
        np.testing.assert_array_equal(imputer.statistics_, ["InternalMedicine"])
        self.assertEqual(
            preprocessor.transform(validation)["medical_specialty"].tolist(),
            ["InternalMedicine", "UnseenSpecialty"],
        )
        if add_indicators:
            self.assertEqual(
                preprocessor.transform(validation)[indicator_name].tolist(), [True, False]
            )
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(validation, original_validation)

    def test_native_categories_and_unseen_values_preserve_inputs_and_model(self):
        model = build_catboost_model(n_jobs=1)
        classifier = model.named_steps["classifier"]
        self.assertFalse(classifier.is_fitted())
        self.assertEqual(list(model.named_steps), ["classifier"])
        self.assertEqual(classifier.get_params()["cat_features"], ["age", "admission_type_id"])
        self.assertEqual(classifier.get_params()["thread_count"], 1)
        self.assertFalse(classifier.get_params()["allow_writing_files"])
        model.set_params(classifier__iterations=3, classifier__depth=2)
        original_train = self.train.copy(deep=True)
        original_validation = self.validation.copy(deep=True)
        model.fit(self.train, self.target)
        fitted_leaf_values = classifier.get_leaf_values().copy()
        probabilities = model.predict_proba(self.validation)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1])
        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertEqual(classifier.feature_names_, list(self.train.columns))
        np.testing.assert_array_equal(classifier.get_leaf_values(), fitted_leaf_values)
        pd.testing.assert_frame_equal(self.train, original_train)
        pd.testing.assert_frame_equal(self.validation, original_validation)

    def test_extended_native_builder_runs_on_each_patient_grouped_fold(self):
        # Synthetic values exercise types, missing markers and unseen diagnosis codes.
        config = FEATURE_SETS["extended"]
        data = pd.DataFrame({
            **{name: [str(i % 4 + 1) for i in range(20)] for name in config["numeric"]},
            **{name: ["?", "recorded"] * 10 for name in config["categorical"]},
            "diag_1": [str(250 + i // 4) for i in range(20)],
            "patient_nbr": [f"SYNTHETIC_{i // 4}" for i in range(20)],
            "readmitted": ["<30", "NO", "NO", "NO"] * 5,
        }, index=range(10, 210, 10))
        original = data.copy(deep=True)
        models = []

        def tiny_builder(feature_set):
            model = build_catboost_model(feature_set=feature_set, n_jobs=1)
            model.set_params(classifier__iterations=3, classifier__depth=2)
            model.fit = Mock(wraps=model.fit)
            models.append(model)
            return model

        results = evaluate_model_cv(data, feature_set="extended", model_builder=tiny_builder)
        self.assertEqual(results["fold"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(np.isfinite(results.to_numpy()).all())
        self.assertEqual(len({id(model) for model in models}), 5)
        X, y = build_model_inputs(data, feature_set="extended")
        for model, (fit_positions, _) in zip(models, make_patient_folds(data)):
            pd.testing.assert_frame_equal(model.fit.call_args.args[0], X.iloc[fit_positions])
            pd.testing.assert_series_equal(model.fit.call_args.args[1], y.iloc[fit_positions])
            self.assertEqual(model.named_steps["classifier"].feature_names_, list(X.columns))
        pd.testing.assert_frame_equal(data, original)


class BaselineModelTest(ModelingFixture):
    def test_builder_does_not_fit_preprocessing_or_classifier(self):
        model = build_baseline_model()
        self.assertFalse(hasattr(model.named_steps["classifier"], "coef_"))
        with self.assertRaises(NotFittedError):
            model.predict_proba(self.validation)

    def test_predict_proba_handles_unseen_category_without_refitting(self):
        model = build_baseline_model()
        model.fit(self.train, self.target)
        preprocessor = model.named_steps["preprocessor"]
        scaler = preprocessor.named_transformers_["numeric"]
        encoder = preprocessor.named_transformers_["categorical"]
        means_before = scaler.mean_.copy()
        categories_before = [values.copy() for values in encoder.categories_]
        coefficients_before = model.named_steps["classifier"].coef_.copy()

        probabilities = model.predict_proba(self.validation)

        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1])
        np.testing.assert_allclose(scaler.mean_, [2.5, 5, 0.5, 0.5])
        np.testing.assert_array_equal(scaler.mean_, means_before)
        for before, after in zip(categories_before, encoder.categories_):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(
            model.named_steps["classifier"].coef_, coefficients_before
        )


class RandomForestModelTest(ModelingFixture):
    def test_forest_is_unfitted_then_reproducible_with_unseen_categories(self):
        # Repeat the small fixture to allow nontrivial leaves at the chosen size.
        train = pd.concat([self.train] * 30, ignore_index=True)
        target = pd.concat([self.target] * 30, ignore_index=True)
        model = build_random_forest_model(n_jobs=1)
        self.assertEqual(model.named_steps["classifier"].n_jobs, 1)
        with self.assertRaises(NotFittedError):
            model.predict_proba(self.validation)
        model.fit(train, target)
        preprocessor = model.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["categorical"]
        numeric_count = 4
        np.testing.assert_allclose(
            preprocessor.transform(train)[:, :numeric_count],
            train.iloc[:, :numeric_count].to_numpy(),
        )
        vocabulary = [values.copy() for values in encoder.categories_]

        probabilities = model.predict_proba(self.validation)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1])
        np.testing.assert_array_equal(model.classes_, [0, 1])
        np.testing.assert_allclose(
            preprocessor.transform(self.validation)[:, :numeric_count],
            self.validation.iloc[:, :numeric_count].to_numpy(),
        )
        for before, after in zip(vocabulary, encoder.categories_):
            np.testing.assert_array_equal(before, after)

        repeated = build_random_forest_model(n_jobs=1)
        repeated.fit(train, target)
        np.testing.assert_allclose(
            repeated.predict_proba(self.validation), probabilities
        )


class LightGBMModelTest(ModelingFixture):
    def test_builder_forwards_requested_imputation_columns(self):
        model = build_lightgbm_model(
            feature_set="extended",
            n_jobs=1,
            imputed_categorical_features=("medical_specialty",),
        )
        branches = {
            name: columns
            for name, _, columns in model.named_steps["preprocessor"].transformers
        }
        self.assertEqual(branches["categorical_imputed"], ["medical_specialty"])
        self.assertNotIn("medical_specialty", branches["categorical"])
        self.assertIn("admission_type_id", branches["categorical"])

    def test_builder_is_unfitted_and_prediction_does_not_refit_preprocessing(self):
        model = build_lightgbm_model(n_jobs=1)
        self.assertEqual(model.named_steps["classifier"].n_jobs, 1)
        with self.assertRaises(NotFittedError):
            model.predict_proba(self.validation)
        # Keep this behavioral test small; production settings stay unchanged.
        model.set_params(
            classifier__n_estimators=3,
            classifier__min_child_samples=1,
            classifier__min_data_in_bin=1,
            classifier__verbosity=-1,
            classifier__force_col_wise=True,
        )
        model.fit(self.train, self.target)
        preprocessor = model.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["categorical"]
        np.testing.assert_allclose(
            preprocessor.transform(self.train)[:, :4], self.train.iloc[:, :4]
        )
        vocabulary = [values.copy() for values in encoder.categories_]
        probabilities = model.predict_proba(self.validation)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1])
        np.testing.assert_array_equal(model.classes_, [0, 1])
        np.testing.assert_allclose(
            preprocessor.transform(self.validation)[:, :4], self.validation.iloc[:, :4]
        )
        for before, after in zip(vocabulary, encoder.categories_):
            np.testing.assert_array_equal(before, after)


class ValidationPredictionTest(ModelingFixture):
    def test_saved_validation_is_only_transformed_and_predictions_keep_row_order(self):
        train = self.train.assign(
            readmitted=["NO", "<30", "NO", "<30"],
            encounter_id=["E1", "E2", "E3", "E4"],
            patient_nbr=["P1", "P2", "P3", "P4"],
        )
        validation = pd.concat([self.validation, self.validation], ignore_index=True)
        validation.index = [900, 800]
        validation["time_in_hospital"] = [14, 13]
        validation["readmitted"] = ["<30", "NO"]
        validation["encounter_id"] = ["E5", "E6"]
        validation["patient_nbr"] = ["P5", "P6"]
        before_train = train.copy(deep=True)
        before_validation = validation.copy(deep=True)
        model = build_baseline_model()
        builder = Mock(return_value=model)

        with patch.object(model, "fit", wraps=model.fit) as fit, patch.object(
            model, "predict_proba", wraps=model.predict_proba
        ) as predict:
            fitted, outcomes, probabilities = fit_and_predict_validation(
                train, validation, builder
            )

        builder.assert_called_once_with(feature_set="initial")
        self.assertIs(fitted, model)
        fit.assert_called_once()
        predict.assert_called_once()
        expected_X, expected_y = build_model_inputs(train)
        expected_validation_X, expected_validation_y = build_model_inputs(validation)
        pd.testing.assert_frame_equal(fit.call_args.args[0], expected_X)
        pd.testing.assert_series_equal(fit.call_args.args[1], expected_y)
        pd.testing.assert_frame_equal(predict.call_args.args[0], expected_validation_X)
        pd.testing.assert_series_equal(outcomes, expected_validation_y)
        self.assertEqual(outcomes.index.tolist(), [900, 800])
        self.assertEqual(probabilities.shape, (2,))
        preprocessor = fitted.named_steps["preprocessor"]
        self.assertEqual(preprocessor.named_transformers_["numeric"].mean_[0], 2.5)
        expected_probabilities = fitted.named_steps["classifier"].predict_proba(
            preprocessor.transform(expected_validation_X)
        )[:, 1]
        np.testing.assert_allclose(probabilities, expected_probabilities)
        pd.testing.assert_frame_equal(train, before_train)
        pd.testing.assert_frame_equal(validation, before_validation)


class ModelCVTest(unittest.TestCase):
    def test_fresh_pipelines_fit_and_score_only_their_fold_rows(self):
        self.check_fold_isolation("initial")

    def test_extended_features_reach_each_fold_preprocessor(self):
        self.check_fold_isolation("extended")

    def test_custom_forest_builder_is_used_in_every_fold(self):
        self.check_fold_isolation("extended", model_kind="forest")

    def test_custom_lightgbm_builder_is_used_in_every_fold(self):
        self.check_fold_isolation("extended", model_kind="lightgbm")

    def check_fold_isolation(self, feature_set, model_kind="logistic"):
        # Different patient prevalences distinguish learned priors from held-out AP.
        train = pd.DataFrame(
            {
                "patient_nbr": [f"P{i // 4}" for i in range(20)],
                "time_in_hospital": [str(i % 14 + 1) for i in range(20)],
                "num_medications": ["3"] * 20,
                "number_emergency": ["0"] * 20,
                "number_inpatient": ["1"] * 20,
                "num_lab_procedures": [str(i + 10) for i in range(20)],
                "num_procedures": [str(i % 3) for i in range(20)],
                "number_outpatient": [str(i % 4) for i in range(20)],
                "number_diagnoses": [str(i % 5 + 1) for i in range(20)],
                "age": ["[50-60)"] * 20,
                "admission_type_id": [str(i // 4) for i in range(20)],
                "admission_source_id": [str(i // 4 + 1) for i in range(20)],
                "max_glu_serum": ["None", "Norm"] * 10,
                "A1Cresult": ["None", ">7", ">8", "Norm"] * 5,
                "race": ["?", "Caucasian"] * 10,
                "gender": ["Female", "Male", "Unknown/Invalid", "Female"] * 5,
                "payer_code": ["MC", "?"] * 10,
                "medical_specialty": ["?", "InternalMedicine"] * 10,
                "diag_1": [str(i // 4 + 250) for i in range(20)],
                "diag_2": ["?", "V45", "250.01", "428"] * 5,
                "diag_3": ["E999", "?"] * 10,
                "readmitted": [
                    "<30" if j < positives else "NO"
                    for positives in [1, 1, 2, 3, 3]
                    for j in range(4)
                ],
            },
            index=range(10, 210, 10),
        )
        for field in MEDICATION_FEATURES:
            train[field] = ["No", "Steady", "Up", "Down"] * 5
        train["change"] = ["No", "Ch"] * 10
        train["diabetesMed"] = ["Yes", "No"] * 10
        original = train.copy(deep=True)
        X, y = build_model_inputs(train, feature_set=feature_set)
        expected_folds = list(make_patient_folds(train))
        models = []
        dummies = []
        requested_feature_sets = []

        def record_new_pipeline(**kwargs):
            requested_feature_sets.append(kwargs["feature_set"])
            if model_kind == "forest":
                model = build_random_forest_model(**kwargs, n_jobs=1)
            elif model_kind == "lightgbm":
                model = build_lightgbm_model(**kwargs, n_jobs=1)
                # Tiny fixtures need smaller leaves/rounds than real training.
                model.set_params(
                    classifier__n_estimators=3,
                    classifier__num_leaves=3,
                    classifier__min_child_samples=1,
                    classifier__min_data_in_bin=1,
                    classifier__verbosity=-1,
                    classifier__force_col_wise=True,
                )
            else:
                model = build_baseline_model(**kwargs)
            model.fit = Mock(wraps=model.fit)
            model.predict_proba = Mock(wraps=model.predict_proba)
            models.append(model)
            return model

        def record_new_dummy(**kwargs):
            dummy = DummyClassifier(**kwargs)
            dummy.fit = Mock(wraps=dummy.fit)
            dummy.predict_proba = Mock(wraps=dummy.predict_proba)
            dummies.append(dummy)
            return dummy

        with patch(
            "carepath_lab.modeling.build_baseline_model", side_effect=record_new_pipeline
        ) as default_builder, patch(
            "carepath_lab.modeling.DummyClassifier", side_effect=record_new_dummy
        ), patch(
            "carepath_lab.modeling.average_precision_score", wraps=average_precision_score
        ) as ap:
            with patch(
                "carepath_lab.modeling.roc_auc_score", wraps=roc_auc_score
            ) as auroc, patch(
                "carepath_lab.modeling.brier_score_loss", wraps=brier_score_loss
            ) as brier:
                if model_kind != "logistic":
                    results = evaluate_model_cv(
                        train, feature_set=feature_set, model_builder=record_new_pipeline
                    )
                    default_builder.assert_not_called()
                elif feature_set == "initial":
                    results = evaluate_model_cv(train)
                else:
                    results = evaluate_model_cv(train, feature_set=feature_set)

        self.assertEqual(len({id(model) for model in models}), 5)
        self.assertEqual(requested_feature_sets, [feature_set] * 5)
        expected_numeric_count = 4 if feature_set == "initial" else 8
        expected_categorical_count = 2 if feature_set == "initial" else 37
        self.assertEqual(X.shape[1], expected_numeric_count + expected_categorical_count)
        self.assertEqual(
            results.columns.tolist(),
            ["fold", "train_ap", "ap", "dummy_ap", "auroc", "dummy_auroc",
             "brier", "dummy_brier"],
        )
        self.assertEqual(results["fold"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(results["ap"].between(0, 1).all())
        self.assertEqual(ap.call_count, 15)
        held_out_ap_calls = [
            call for index, call in enumerate(ap.call_args_list) if index % 3 != 0
        ]
        # AUROC/Brier score held-out rows only; training AP is a separate diagnostic.
        for metric in [auroc, brier]:
            self.assertEqual(metric.call_count, 10)
            for metric_call, ap_call in zip(metric.call_args_list, held_out_ap_calls):
                pd.testing.assert_series_equal(metric_call.args[0], ap_call.args[0])
                np.testing.assert_allclose(metric_call.args[1], ap_call.args[1])
        for fold_index in range(5):
            for name, function, metric in [
                ("auroc", roc_auc_score, auroc),
                ("brier", brier_score_loss, brier),
            ]:
                for offset, prefix in [(0, ""), (1, "dummy_")]:
                    call = metric.call_args_list[2 * fold_index + offset]
                    self.assertAlmostEqual(
                        results.iloc[fold_index][prefix + name], function(*call.args)
                    )
        np.testing.assert_allclose(results["dummy_auroc"], 0.5)
        for model, (fit_positions, validation_positions), score_call in zip(
            models, expected_folds, ap.call_args_list[1::3]
        ):
            model.fit.assert_called_once()
            fit_X, fit_y = model.fit.call_args.args
            pd.testing.assert_frame_equal(fit_X, X.iloc[fit_positions])
            pd.testing.assert_series_equal(fit_y, y.iloc[fit_positions])
            self.assertEqual(model.predict_proba.call_count, 2)
            pd.testing.assert_frame_equal(
                model.predict_proba.call_args_list[0].args[0], X.iloc[validation_positions]
            )
            pd.testing.assert_frame_equal(
                model.predict_proba.call_args_list[1].args[0], X.iloc[fit_positions]
            )
            pd.testing.assert_series_equal(
                score_call.args[0], y.iloc[validation_positions]
            )
            self.assertEqual(score_call.args[1].shape, (len(validation_positions),))
            expected_scores = model.predict_proba(X.iloc[validation_positions])[:, 1]
            np.testing.assert_allclose(score_call.args[1], expected_scores)

            preprocessor = model.named_steps["preprocessor"]
            self.assertEqual(
                issparse(preprocessor.transform(X.iloc[validation_positions])),
                feature_set == "extended",
            )
            if model_kind == "logistic":
                scaler = preprocessor.named_transformers_["numeric"]
                self.assertEqual(scaler.n_features_in_, expected_numeric_count)
                np.testing.assert_allclose(
                    scaler.mean_,
                    X.iloc[fit_positions].select_dtypes(include="number").mean().to_numpy(),
                )
            else:
                transformed = preprocessor.transform(X.iloc[validation_positions])
                if issparse(transformed):
                    transformed = transformed.toarray()
                np.testing.assert_allclose(
                    transformed[:, :expected_numeric_count],
                    X.iloc[validation_positions, :expected_numeric_count],
                )
            encoder = preprocessor.named_transformers_["categorical"]
            self.assertEqual(
                set(encoder.categories_[1]),
                set(X.iloc[fit_positions]["admission_type_id"]),
            )
            if feature_set == "extended":
                self.assertEqual(
                    set(encoder.categories_[2]),
                    set(X.iloc[fit_positions]["admission_source_id"]),
                )
                for offset, field in enumerate(["max_glu_serum", "A1Cresult"], start=3):
                    self.assertEqual(
                        set(encoder.categories_[offset]), set(X.iloc[fit_positions][field])
                    )
                for offset, field in enumerate(
                    [*MEDICATION_FEATURES, "change", "diabetesMed",
                     "race", "gender", "payer_code", "medical_specialty",
                     "diag_1", "diag_2", "diag_3"], start=5
                ):
                    self.assertEqual(
                        set(encoder.categories_[offset]), set(X.iloc[fit_positions][field])
                    )

        self.assertEqual(len({id(dummy) for dummy in dummies}), 5)
        for row, dummy, (fit_positions, validation_positions), score_call in zip(
            results.itertuples(), dummies, expected_folds, ap.call_args_list[2::3]
        ):
            self.assertEqual(dummy.strategy, "prior")
            dummy.fit.assert_called_once()
            pd.testing.assert_frame_equal(dummy.fit.call_args.args[0], X.iloc[fit_positions])
            pd.testing.assert_series_equal(dummy.fit.call_args.args[1], y.iloc[fit_positions])
            dummy.predict_proba.assert_called_once()
            pd.testing.assert_frame_equal(
                dummy.predict_proba.call_args.args[0], X.iloc[validation_positions]
            )
            pd.testing.assert_series_equal(score_call.args[0], y.iloc[validation_positions])
            self.assertEqual(score_call.args[1].shape, (len(validation_positions),))
            np.testing.assert_allclose(score_call.args[1], y.iloc[fit_positions].mean())
            self.assertAlmostEqual(row.dummy_ap, y.iloc[validation_positions].mean())
        for row, model, (fit_positions, _), training_call in zip(
            results.itertuples(), models, expected_folds, ap.call_args_list[::3]
        ):
            pd.testing.assert_series_equal(training_call.args[0], y.iloc[fit_positions])
            self.assertEqual(training_call.args[1].shape, (len(fit_positions),))
            np.testing.assert_allclose(
                training_call.args[1], model.predict_proba(X.iloc[fit_positions])[:, 1]
            )
            self.assertAlmostEqual(row.train_ap, average_precision_score(*training_call.args))
        self.assertTrue(
            any(
                y.iloc[fit].mean() != y.iloc[held_out].mean()
                for fit, held_out in expected_folds
            )
        )
        pd.testing.assert_frame_equal(train, original)
