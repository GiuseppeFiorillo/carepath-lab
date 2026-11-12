import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import evaluate_models


class EvaluateModelsCommandTest(unittest.TestCase):
    def test_all_models_use_same_saved_partitions_scores_and_calibration_rows(self):
        train = pd.DataFrame({"patient_nbr": ["TRAIN_A", "TRAIN_B"]})
        validation = pd.DataFrame(
            {"patient_nbr": ["HOLDOUT_A", "HOLDOUT_A"]}, index=[90, 10]
        )
        outcomes = pd.Series([0, 1], index=validation.index, name="readmitted")
        probabilities = [np.array([0.1, 0.8]), np.array([0.3, 0.7]), np.array([0.2, 0.6])]
        bootstrap_result = pd.DataFrame(
            {"logistic": [0.02, 0.03], "random_forest": [0.08, 0.09],
             "lightgbm": [0.07, 0.12]}
        )
        ap_bootstrap_result = pd.DataFrame(
            {"logistic": [0.3, 0.4], "random_forest": [0.5, 0.9],
             "lightgbm": [0.6, 0.7]}
        )
        output = io.StringIO()

        with patch(
            "scripts.evaluate_models.load_verified_partitions",
            return_value=(train, validation),
        ), patch(
            "scripts.evaluate_models.fit_and_predict_validation",
            side_effect=[(object(), outcomes, values) for values in probabilities],
        ) as fit, patch(
            "scripts.evaluate_models.print_calibration_summary"
        ) as calibration, patch(
            "scripts.evaluate_models.bootstrap_metric_scores",
            side_effect=[bootstrap_result, ap_bootstrap_result],
        ) as bootstrap, contextlib.redirect_stdout(output):
            evaluate_models.main()

        expected_builders = [
            evaluate_models.build_baseline_model,
            evaluate_models.build_random_forest_model,
            evaluate_models.build_lightgbm_model,
        ]
        self.assertEqual(fit.call_count, 3)
        self.assertEqual(calibration.call_count, 3)
        self.assertEqual(bootstrap.call_count, 2)
        for call in bootstrap.call_args_list:
            pd.testing.assert_series_equal(call.args[0], outcomes)
            self.assertEqual(
                list(call.args[1]), ["logistic", "random_forest", "lightgbm"]
            )
            for values, supplied in zip(probabilities, call.args[1].values()):
                np.testing.assert_array_equal(values, supplied)
            pd.testing.assert_series_equal(call.args[2], validation["patient_nbr"])
        self.assertEqual(bootstrap.call_args_list[0].kwargs, {})
        self.assertIs(
            bootstrap.call_args_list[1].kwargs["score_function"],
            evaluate_models.average_precision_score,
        )
        self.assertIn("Bootstrap Brier scores: (2, 3)", output.getvalue())
        self.assertIn("Bootstrap AP scores: (2, 3)", output.getvalue())
        # AP uses its own table, with paired differences [-0.1, 0.2].
        self.assertIn(
            "AP difference (RF - LightGBM): 0.000000\n"
            "95% patient-cluster bootstrap percentile CI: [-0.092500, 0.192500]",
            output.getvalue(),
        )
        # Quantiles of paired differences [-0.01, 0.03], not marginal endpoints.
        for pair, point, lower, upper in [
            ("lightgbm - random_forest", "0.010000", "-0.009000", "0.029000"),
            ("lightgbm - logistic", "0.075000", "0.051000", "0.089000"),
            ("random_forest - logistic", "0.065000", "0.060000", "0.060000"),
        ]:
            self.assertIn(
                f"Brier difference ({pair}): {point}\n"
                "95% patient-cluster bootstrap percentile CI: "
                f"[{lower}, {upper}]",
                output.getvalue(),
            )
        for call, builder, calibration_call, values in zip(
            fit.call_args_list, expected_builders, calibration.call_args_list, probabilities
        ):
            self.assertIs(call.args[0], train)
            self.assertIs(call.args[1], validation)
            self.assertIs(call.args[2], builder)
            self.assertEqual(call.kwargs, {"feature_set": "extended"})
            pd.testing.assert_series_equal(calibration_call.args[0], outcomes)
            np.testing.assert_array_equal(calibration_call.args[1], values)
            pd.testing.assert_series_equal(
                calibration_call.args[2], validation["patient_nbr"]
            )
        # Actual metrics on synthetic vectors, not model-performance assertions.
        self.assertRegex(output.getvalue(), r"logistic\s+2\s+1\.000000\s+1\.000000\s+0\.025000")
        self.assertRegex(output.getvalue(), r"random_forest\s+2\s+1\.000000\s+1\.000000\s+0\.090000")
        self.assertRegex(output.getvalue(), r"lightgbm\s+2\s+1\.000000\s+1\.000000\s+0\.100000")
