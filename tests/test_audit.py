import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from carepath_lab.audit import (
    build_audit_table,
    bootstrap_gender_recall_difference,
    summarize_subgroup_calibration,
    summarize_subgroup_metrics,
)


class BuildAuditTableTest(unittest.TestCase):
    def setUp(self):
        self.validation = pd.DataFrame({
            "encounter_id": ["001", "002", "003"],
            "patient_nbr": ["P1", "P1", "P2"],
            "age": ["[50-60)"] * 3,
            "gender": ["Female", "Female", "Male"],
            "race": ["?"] * 3,
            "readmitted": ["<30", ">30", "NO"],
        }, index=[10, 20, 30])
        self.scores = pd.DataFrame({
            "encounter_id": ["003", "001", "002"],
            "patient_nbr": ["P2", "P1", "P1"],
            "outcome": [0, 1, 0],
            "catboost_specialty_mode": [0.05, 0.30, 0.10],
        }, index=[70, 80, 90])

    def test_aligns_by_key_despite_different_order_without_mutating_inputs(self):
        validation_before = self.validation.copy(deep=True)
        scores_before = self.scores.copy(deep=True)
        result = build_audit_table(self.validation, self.scores)
        by_encounter = result.set_index("encounter_id")
        self.assertEqual(len(result), 3)
        self.assertTrue(result["_merge"].eq("both").all())
        self.assertEqual(by_encounter.loc["001", "outcome"], 1)
        self.assertEqual(by_encounter.loc["001", "catboost_specialty_mode"], 0.30)
        self.assertTrue(result["patient_nbr_source"].eq(result["patient_nbr_saved"]).all())
        pd.testing.assert_frame_equal(self.validation, validation_before)
        pd.testing.assert_frame_equal(self.scores, scores_before)

    def test_missing_and_extra_prediction_keys_are_rejected(self):
        extra = self.scores.iloc[[0]].copy()
        extra["encounter_id"] = "EXTRA"
        cases = [self.scores.iloc[1:], pd.concat([self.scores, extra])]
        for scores in cases:
            with self.subTest(keys=scores["encounter_id"].tolist()):
                with self.assertRaisesRegex(ValueError, "encounter IDs"):
                    build_audit_table(self.validation, scores)

    def test_duplicate_keys_on_either_side_are_rejected(self):
        for side in ["validation", "scores"]:
            with self.subTest(side=side):
                validation, scores = self.validation, self.scores
                if side == "validation":
                    validation = pd.concat([validation, validation.iloc[[0]]])
                else:
                    scores = pd.concat([scores, scores.iloc[[0]]])
                with self.assertRaises(pd.errors.MergeError):
                    build_audit_table(validation, scores)

    def test_patient_disagreement_is_rejected(self):
        self.scores.loc[70, "patient_nbr"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "Patient IDs"):
            build_audit_table(self.validation, self.scores)

    def test_swapped_outcomes_with_same_positive_count_are_rejected(self):
        self.scores.loc[[70, 80], "outcome"] = [1, 0]
        self.assertEqual(self.scores["outcome"].sum(), 1)
        with self.assertRaisesRegex(ValueError, "Recorded outcomes"):
            build_audit_table(self.validation, self.scores)


class SummarizeSubgroupMetricsTest(unittest.TestCase):
    def setUp(self):
        self.audit = pd.DataFrame({
            "gender": ["Female"] * 4 + ["Unknown/Invalid"],
            "age": ["[50-60)"] * 5,
            "race": ["?"] * 5,
            "patient_nbr_source": ["P1", "P1", "P2", "P3", "P4"],
            "outcome": [1, 0, 1, 0, 0],
            "catboost_specialty_mode": [0.20, 0.30, 0.05, 0.01, 0.90],
        }, index=[20, 40, 60, 80, 100])

    def test_counts_threshold_boundary_support_and_input_preservation(self):
        before = self.audit.copy(deep=True)
        result = summarize_subgroup_metrics(self.audit)
        row = result.loc[
            result["group_column"].eq("gender")
            & result["group_value"].eq("Female")
            & result["threshold"].eq(0.20)
        ].iloc[0]

        self.assertEqual(result.shape, (16, 13))
        self.assertEqual(row[["tp", "fn", "fp", "tn"]].tolist(), [1, 1, 1, 1])
        self.assertEqual(row["encounter_count"], 4)
        self.assertEqual(row["patient_count"], 3)
        self.assertEqual(row["positive_count"], 2)
        for metric in ["recall", "precision", "fpr"]:
            self.assertEqual(row[metric], 0.5)
        totals = result.groupby(["group_column", "threshold"])[
            ["encounter_count", "positive_count"]
        ].sum()
        self.assertTrue(totals["encounter_count"].eq(5).all())
        self.assertTrue(totals["positive_count"].eq(2).all())
        pd.testing.assert_frame_equal(self.audit, before)

    def test_no_positives_and_no_positive_predictions_are_distinct(self):
        result = summarize_subgroup_metrics(self.audit)
        gender = result.loc[result["group_column"].eq("gender")]
        unknown = gender.loc[gender["group_value"].eq("Unknown/Invalid")]
        self.assertTrue(unknown["recall"].isna().all())
        self.assertTrue(unknown["precision"].eq(0).all())
        self.assertTrue(unknown["fpr"].eq(1).all())

        female = gender.loc[
            gender["group_value"].eq("Female") & gender["threshold"].eq(0.50)
        ].iloc[0]
        self.assertEqual(female["recall"], 0)
        self.assertTrue(pd.isna(female["precision"]))
        self.assertEqual(female["fpr"], 0)

    def test_no_real_negatives_yields_undefined_fpr_and_missing_group_is_kept(self):
        self.audit["outcome"] = 1
        self.audit["race"] = None
        result = summarize_subgroup_metrics(self.audit)
        self.assertTrue(result["fpr"].isna().all())
        missing_race = result.loc[result["group_column"].eq("race")]
        self.assertEqual(len(missing_race), 4)
        self.assertTrue(missing_race["group_value"].isna().all())
        self.assertTrue(missing_race["encounter_count"].eq(5).all())


    def test_invalid_probabilities_and_outcomes_are_rejected(self):
        invalid_values = {
            "catboost_specialty_mode": [
                float("nan"), float("inf"), -0.01, 1.01,
            ],
            "outcome": [float("nan"), -1, 2],
        }

        for column, values in invalid_values.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    invalid = self.audit.copy()
                    invalid.loc[invalid.index[0], column] = value

                    with self.assertRaises(ValueError):
                        summarize_subgroup_metrics(invalid)

class SummarizeSubgroupCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.audit = pd.DataFrame({
            "gender": ["Female", "Female", "Female", "Male"],
            "age": ["[50-60)"] * 4,
            "race": [None] * 4,
            "patient_nbr_source": ["P1", "P1", "P2", "P3"],
            "outcome": [0, 1, 0, 1],
            "catboost_specialty_mode": [0.0, 0.10, 0.20, 1.0],
        }, index=[10, 30, 50, 70])

    def test_bin_boundaries_means_support_and_input_preservation(self):
        before = self.audit.copy(deep=True)
        result = summarize_subgroup_calibration(self.audit)
        female = result.loc[
            result["group_column"].eq("gender")
            & result["group_value"].eq("Female")
        ]
        low = female.loc[female["probability_bin"].map(lambda x: x.right).eq(0.1)].iloc[0]
        self.assertEqual(low["encounter_count"], 2)
        self.assertEqual(low["patient_count"], 1)
        self.assertEqual(low["positive_count"], 1)
        self.assertAlmostEqual(low["mean_predicted_probability"], 0.05)
        self.assertEqual(low["observed_positive_fraction"], 0.5)
        self.assertAlmostEqual(low["prediction_minus_observed"], -0.45)
        middle = female.loc[
            female["probability_bin"].map(lambda x: x.right).eq(0.2)
        ].iloc[0]
        self.assertEqual(middle["encounter_count"], 1)
        self.assertAlmostEqual(middle["prediction_minus_observed"], 0.2)
        self.assertEqual(len(female), 2)  # Empty bins must not appear as observations.
        male = result.loc[
            result["group_column"].eq("gender") & result["group_value"].eq("Male")
        ].iloc[0]
        self.assertEqual(male["probability_bin"].right, 1.0)
        self.assertEqual(male["mean_predicted_probability"], 1.0)
        pd.testing.assert_frame_equal(self.audit, before)

    def test_totals_and_missing_demographic_group_are_preserved(self):
        result = summarize_subgroup_calibration(self.audit)
        totals = result.groupby("group_column")[["encounter_count", "positive_count"]].sum()
        self.assertTrue(totals["encounter_count"].eq(4).all())
        self.assertTrue(totals["positive_count"].eq(2).all())
        race = result.loc[result["group_column"].eq("race")]
        self.assertTrue(race["group_value"].isna().all())
        self.assertTrue(result["encounter_count"].gt(0).all())

    def test_invalid_probabilities_and_outcomes_are_rejected(self):
        for column, values in {
            "catboost_specialty_mode": [float("nan"), float("inf"), -0.01, 1.01],
            "outcome": [float("nan"), -1, 2],
        }.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    invalid = self.audit.copy()
                    invalid.loc[invalid.index[0], column] = value
                    with self.assertRaises(ValueError):
                        summarize_subgroup_calibration(invalid)


class BootstrapGenderRecallDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.audit = pd.DataFrame({
            "gender": ["Female", "Female", "Male", "Male"],
            "age": ["[50-60)"] * 4,
            "race": ["?"] * 4,
            "patient_nbr_source": ["P1", "P1", "P2", "P3"],
            "outcome": [1, 1, 1, 1],
            "catboost_specialty_mode": [0.30, 0.05, 0.30, 0.05],
        }, index=[10, 30, 50, 70])

    def test_shared_draws_keep_patient_multiplicity_and_difference_direction(self):
        before = self.audit.copy(deep=True)
        with patch(
            "carepath_lab.audit.sample_patient_positions",
            side_effect=[[0, 1, 0, 1, 2], [0, 1, 3]],
        ) as sample:
            result = bootstrap_gender_recall_difference(self.audit, n_bootstrap=2)
        np.testing.assert_allclose(result, [-0.5, 0.5])
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(sample.call_args_list[0].args[0], {"P1": [0, 1], "P2": [2], "P3": [3]})
        self.assertIs(sample.call_args_list[0].args[1], sample.call_args_list[1].args[1])
        pd.testing.assert_frame_equal(self.audit, before)

    def test_absent_groups_and_no_real_positives_remain_nan(self):
        self.audit.loc[50, "outcome"] = 0
        with patch(
            "carepath_lab.audit.sample_patient_positions",
            side_effect=[[0, 1], [2], [0, 1, 2]],
        ):
            result = bootstrap_gender_recall_difference(self.audit, n_bootstrap=3)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.isnan(result).all())

    def test_requested_seed_is_used_and_repeated_calls_are_reproducible(self):
        with patch(
            "carepath_lab.audit.np.random.default_rng", wraps=np.random.default_rng
        ) as rng:
            first = bootstrap_gender_recall_difference(self.audit, n_bootstrap=5, random_state=321)
            repeated = bootstrap_gender_recall_difference(self.audit, n_bootstrap=5, random_state=321)
        self.assertEqual(rng.call_count, 2)
        rng.assert_called_with(321)
        np.testing.assert_allclose(first, repeated, equal_nan=True)

    def test_nonpositive_replication_count_is_rejected(self):
        for count in [0, -1]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    bootstrap_gender_recall_difference(self.audit, n_bootstrap=count)


if __name__ == "__main__":
    unittest.main()
