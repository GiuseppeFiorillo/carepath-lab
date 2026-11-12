from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from carepath_lab.uncertainty import (
    bootstrap_metric_scores,
    group_positions_by_patient,
    sample_patient_positions,
)


class GroupPositionsTest(unittest.TestCase):
    def test_repeated_patients_keep_all_encounter_positions(self):
        patients = ["P1", "P2", "P1", "P3", "P2"]
        result = group_positions_by_patient(patients)
        self.assertEqual(result, {"P1": [0, 2], "P2": [1, 4], "P3": [3]})

    def test_series_index_labels_do_not_replace_positions(self):
        patients = pd.Series(["P1", "P2", "P1"], index=[70, 10, 30])
        before = patients.copy()
        result = group_positions_by_patient(patients)
        self.assertEqual(result, {"P1": [0, 2], "P2": [1]})
        pd.testing.assert_series_equal(patients, before)


class SamplePatientPositionsTest(unittest.TestCase):
    def test_repeated_draws_keep_whole_blocks_and_multiplicity(self):
        positions = {"P1": [0, 1, 2], "P2": [3]}
        before = deepcopy(positions)
        rng = Mock()
        rng.choice.return_value = np.array(["P1", "P1"])

        sampled = sample_patient_positions(positions, rng)

        self.assertEqual(sampled, [0, 1, 2, 0, 1, 2])
        rng.choice.assert_called_once_with(["P1", "P2"], size=2, replace=True)
        self.assertEqual(positions, before)

    def test_caller_rng_advances_and_reproduces_the_sequence_with_same_seed(self):
        positions = {"P1": [0, 1], "P2": [2], "P3": [3, 4, 5]}
        rng = np.random.default_rng(123)
        before = deepcopy(rng.bit_generator.state)
        sequence = [sample_patient_positions(positions, rng) for _ in range(2)]
        self.assertNotEqual(before, rng.bit_generator.state)

        repeated_rng = np.random.default_rng(123)
        repeated = [
            sample_patient_positions(positions, repeated_rng) for _ in range(2)
        ]
        self.assertEqual(sequence, repeated)


class BootstrapMetricScoresTest(unittest.TestCase):
    def setUp(self):
        self.outcomes = pd.Series([0, 1, 1], index=[90, 20, 70])
        self.patients = pd.Series(["P1", "P2", "P1"], index=self.outcomes.index)
        self.predictions = {
            "logistic": np.array([0.1, 0.8, 0.4]),
            "random_forest": np.array([0.2, 0.7, 0.8]),
            "lightgbm": np.array([0.3, 0.9, 0.6]),
        }

    def test_identical_patient_draws_score_all_models_and_keep_repetitions(self):
        draws = [[0, 2, 0, 2], [1, 1]]  # P1 twice, then P2 twice.
        with patch(
            "carepath_lab.uncertainty.sample_patient_positions", side_effect=draws
        ) as sample, patch(
            "carepath_lab.uncertainty.brier_score_loss", wraps=brier_score_loss
        ) as metric:
            result = bootstrap_metric_scores(
                self.outcomes, self.predictions, self.patients, n_bootstrap=2
            )

        self.assertEqual(sample.call_count, 2)
        self.assertIs(sample.call_args_list[0].args[1], sample.call_args_list[1].args[1])
        self.assertEqual(sample.call_args_list[0].args[0], {"P1": [0, 2], "P2": [1]})
        self.assertEqual(metric.call_count, 6)
        expected = pd.DataFrame(
            {"logistic": [0.185, 0.04], "random_forest": [0.04, 0.09],
             "lightgbm": [0.125, 0.01]}
        )
        pd.testing.assert_frame_equal(result, expected)
        for replicate, positions in enumerate(draws):
            for offset, values in enumerate(self.predictions.values()):
                call = metric.call_args_list[3 * replicate + offset]
                pd.testing.assert_series_equal(call.args[0], self.outcomes.iloc[positions])
                np.testing.assert_array_equal(call.args[1], values[positions])

    def test_seeded_results_are_reproducible_without_mutating_inputs(self):
        original_outcomes = self.outcomes.copy(deep=True)
        original_patients = self.patients.copy(deep=True)
        original_predictions = deepcopy(self.predictions)
        first = bootstrap_metric_scores(
            self.outcomes, self.predictions, self.patients, n_bootstrap=5, random_state=123
        )
        repeated = bootstrap_metric_scores(
            self.outcomes, self.predictions, self.patients, n_bootstrap=5, random_state=123
        )
        self.assertEqual(first.shape, (5, 3))
        pd.testing.assert_frame_equal(first, repeated)
        pd.testing.assert_series_equal(self.outcomes, original_outcomes)
        pd.testing.assert_series_equal(self.patients, original_patients)
        for name, original in original_predictions.items():
            np.testing.assert_array_equal(self.predictions[name], original)

    def test_custom_ap_scoring_keeps_shared_draws_and_positive_scores(self):
        predictions = deepcopy(self.predictions)
        predictions["logistic"] = np.array([0.9, 0.8, 0.4])
        metric = Mock(wraps=average_precision_score)
        draws = [[0, 2, 0, 2], [1, 1]]
        with patch(
            "carepath_lab.uncertainty.sample_patient_positions", side_effect=draws
        ) as sample:
            result = bootstrap_metric_scores(
                self.outcomes, predictions, self.patients, n_bootstrap=2,
                score_function=metric,
            )
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(metric.call_count, 6)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"logistic": [0.5, 1.0], "random_forest": [1.0, 1.0],
                          "lightgbm": [1.0, 1.0]}),
        )
        for replicate, positions in enumerate(draws):
            for offset, values in enumerate(predictions.values()):
                call = metric.call_args_list[3 * replicate + offset]
                pd.testing.assert_series_equal(call.args[0], self.outcomes.iloc[positions])
                np.testing.assert_array_equal(call.args[1], values[positions])

    def test_invalid_metric_results_raise_instead_of_disappearing_from_quantiles(self):
        for score in [np.nan, np.inf, -np.inf, np.array([0.1, 0.2])]:
            with self.subTest(score=score):
                metric = Mock(return_value=score)
                with self.assertRaisesRegex(ValueError, "finite scalar"):
                    bootstrap_metric_scores(
                        self.outcomes, self.predictions, self.patients,
                        n_bootstrap=2, score_function=metric,
                    )

    def test_invalid_alignment_shapes_and_empty_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "indices"):
            bootstrap_metric_scores(
                self.outcomes, self.predictions, self.patients.iloc[::-1], n_bootstrap=2
            )
        for values in [np.array([0.1, 0.2]), np.array([[0.1], [0.2], [0.3]])]:
            with self.subTest(shape=values.shape):
                with self.assertRaisesRegex(ValueError, "Prediction vector"):
                    bootstrap_metric_scores(
                        self.outcomes, {"bad": values}, self.patients, n_bootstrap=2
                    )
        for count in [0, -1]:
            with self.subTest(n_bootstrap=count):
                with self.assertRaisesRegex(ValueError, "positive"):
                    bootstrap_metric_scores(
                        self.outcomes, self.predictions, self.patients, n_bootstrap=count
                    )
        with self.assertRaisesRegex(ValueError, "At least one"):
            bootstrap_metric_scores(self.outcomes, {}, self.patients, n_bootstrap=2)
        with self.assertRaisesRegex(ValueError, "At least one"):
            bootstrap_metric_scores(
                self.outcomes.iloc[:0], {"empty": np.array([])},
                self.patients.iloc[:0], n_bootstrap=2,
            )
