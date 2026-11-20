from copy import deepcopy
import unittest

import pandas as pd

from carepath_lab.features import FEATURE_SETS
from scripts.review_features import make_review_profiles


class ReviewProfilesTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({name: ["No", "No"] for name in ["acetohexamide", "examide", "citoglipton"]})

    def test_declared_exclusions_preserve_reference_and_feature_order(self):
        before = deepcopy(FEATURE_SETS)
        profiles = make_review_profiles(self.train)
        self.assertEqual(len(profiles), 8)
        reference = profiles["reference"]
        self.assertEqual(reference, FEATURE_SETS["extended"])
        cleaned = profiles["no_constants"]
        self.assertEqual(len(cleaned["numeric"]) + len(cleaned["categorical"]), 42)
        self.assertNotIn("payer_code", profiles["no_payer"]["categorical"])
        self.assertNotIn("medical_specialty", profiles["no_specialty"]["categorical"])
        self.assertNotIn("A1Cresult", profiles["no_labs"]["categorical"])
        self.assertNotIn("diag_3", profiles["no_diagnoses"]["categorical"])
        self.assertNotIn("race", profiles["no_demographics"]["categorical"])
        for config in profiles.values():
            ordered = config["numeric"] + config["categorical"]
            self.assertEqual(len(ordered), len(set(ordered)))
            self.assertFalse({"patient_nbr", "encounter_id", "readmitted", "weight", "discharge_disposition_id"} & set(ordered))
        self.assertEqual(FEATURE_SETS, before)

    def test_declared_constant_exclusion_is_rejected_if_training_varies(self):
        self.train.loc[1, "examide"] = "Steady"
        with self.assertRaisesRegex(ValueError, "constant"):
            make_review_profiles(self.train)


if __name__ == "__main__":
    unittest.main()
