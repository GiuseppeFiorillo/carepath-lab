"""Check search orchestration without fitting models or reading real data."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import tune_lightgbm
from carepath_lab.features import FEATURE_SETS


class LightGBMSearchTest(unittest.TestCase):
    def test_main_rejects_existing_output_before_loading_or_searching(self):
        with TemporaryDirectory() as temporary_dir:
            with (
                patch("scripts.tune_lightgbm.Path", return_value=Path(temporary_dir)),
                patch("scripts.tune_lightgbm.load_verified_partitions") as load,
                patch("scripts.tune_lightgbm.run_search") as search,
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(FileExistsError):
                    tune_lightgbm.main()
            load.assert_not_called()
            search.assert_not_called()

    def test_main_passes_only_training_and_exports_before_hashing(self):
        train = pd.DataFrame({"patient_nbr": ["SYNTHETIC_TRAIN"]})
        validation = pd.DataFrame({"patient_nbr": ["SYNTHETIC_VALIDATION"]})
        scores = pd.DataFrame({
            "configuration_id": [1, 1], "num_leaves": [15, 15],
            "min_child_samples": [20, 20], "reg_lambda": [0.0, 0.0],
            "fold": [1, 2], "ap": [0.1, 0.2], "train_ap": [0.3, 0.4],
            "auroc": [0.5, 0.6], "brier": [0.2, 0.3],
        })
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "new_run"

            def check_metadata_inputs(configurations, destination):
                self.assertEqual(destination, output_dir)
                self.assertEqual(len(configurations), 8)
                self.assertTrue((destination / "fold_results.csv").is_file())
                self.assertTrue((destination / "summary.csv").is_file())
                return {"synthetic_test_metadata": True}

            with (
                patch("scripts.tune_lightgbm.Path", return_value=output_dir),
                patch("scripts.tune_lightgbm.load_verified_partitions", return_value=(train, validation)),
                patch("scripts.tune_lightgbm.run_search", return_value=scores) as search,
                patch("scripts.tune_lightgbm.build_metadata", side_effect=check_metadata_inputs),
                redirect_stdout(StringIO()),
            ):
                tune_lightgbm.main()

            search.assert_called_once()
            self.assertIs(search.call_args.args[0], train)
            self.assertEqual(len(search.call_args.args[1]), 8)
            pd.testing.assert_frame_equal(pd.read_csv(output_dir / "fold_results.csv"), scores)
            pd.testing.assert_frame_equal(
                pd.read_csv(output_dir / "summary.csv"),
                tune_lightgbm.summarize_search(scores).reset_index(drop=True),
            )
            self.assertEqual(
                json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")),
                {"synthetic_test_metadata": True},
            )

    def test_metadata_writer_roundtrips_utf8_and_refuses_overwrite(self):
        metadata = {"description": "probabilità", "cv": {"shuffle": True}}
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            tune_lightgbm.save_metadata(metadata, output_dir)
            path = output_dir / "metadata.json"
            with path.open(encoding="utf-8") as file:
                self.assertEqual(json.load(file), metadata)
            before = path.read_bytes()
            self.assertIn("probabilità".encode("utf-8"), before)
            self.assertTrue(before.endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                tune_lightgbm.save_metadata({"replacement": True}, output_dir)
            self.assertEqual(path.read_bytes(), before)

    def test_metadata_records_protocol_and_hashes_requested_output_files(self):
        configurations = [
            {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 10.0}
        ]
        output_dir = Path("synthetic_outputs") / "run"
        paths = [
            Path("data/raw/diabetic_data.csv"),
            Path("data/processed/split_assignments.csv"),
            Path("data/processed/split_metadata.json"),
            output_dir / "fold_results.csv",
            output_dir / "summary.csv",
        ]
        synthetic_hashes = {str(path): str(index) * 64 for index, path in enumerate(paths)}
        with patch(
            "scripts.tune_lightgbm.file_sha256",
            side_effect=lambda path: synthetic_hashes[str(path)],
        ) as hash_file:
            metadata = tune_lightgbm.build_metadata(configurations, output_dir)

        self.assertEqual([call.args[0] for call in hash_file.call_args_list], paths)
        self.assertEqual(metadata["file_sha256"], synthetic_hashes)
        self.assertEqual(metadata["evaluation_partition"], "train")
        self.assertEqual(metadata["selection_metric"], "mean_ap")
        self.assertEqual(metadata["selection_direction"], "maximize")
        self.assertEqual(metadata["configurations"], configurations)
        self.assertEqual(metadata["features"], FEATURE_SETS["extended"])
        self.assertEqual(metadata["imputed_categorical_features"], [])
        self.assertEqual(metadata["preprocessing"], {
            "scale_numeric": False, "categorical_encoding": "onehot",
        })
        self.assertEqual(metadata["cv"], {
            "method": "GroupKFold", "group_column": "patient_nbr",
            "n_splits": 5, "shuffle": True, "random_state": 42,
        })
        self.assertEqual(set(metadata["environment"]), {
            "python", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm",
        })
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)
        self.assertIsNot(metadata["features"]["numeric"], FEATURE_SETS["extended"]["numeric"])

    def test_file_hash_uses_contents_not_filename(self):
        with TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "original.bin"
            path.write_bytes(b"abc")
            expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(tune_lightgbm.file_sha256(path), expected)
            renamed = path.rename(path.with_name("renamed.bin"))
            self.assertEqual(tune_lightgbm.file_sha256(renamed), expected)
            renamed.write_bytes(b"abd")
            self.assertNotEqual(tune_lightgbm.file_sha256(renamed), expected)

    def test_csv_writer_preserves_values_and_uses_requested_directory(self):
        results = pd.DataFrame({"fold": [1, 2], "ap": [0.1234567890123, 0.2]})
        summary = pd.DataFrame({"mean_ap": [0.16172839450615]})
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "nested" / "run"
            tune_lightgbm.save_csv_results(results, summary, output_dir)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"fold_results.csv", "summary.csv"},
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(output_dir / "fold_results.csv"), results,
                check_exact=False, rtol=1e-14, atol=0,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(output_dir / "summary.csv"), summary,
                check_exact=False, rtol=1e-14, atol=0,
            )
            before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
            with self.assertRaises(FileExistsError):
                tune_lightgbm.save_csv_results(results * 0, summary * 0, output_dir)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output_dir.iterdir()}, before
            )

    def test_csv_writer_rejects_even_an_empty_existing_directory(self):
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with self.assertRaises(FileExistsError):
                tune_lightgbm.save_csv_results(pd.DataFrame(), pd.DataFrame(), output_dir)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_summary_aggregates_each_configuration_and_ranks_by_held_out_ap(self):
        results = pd.DataFrame(
            {
                "configuration_id": [2, 1, 2, 1],
                "num_leaves": [31, 15, 31, 15],
                "min_child_samples": [100, 20, 100, 20],
                "reg_lambda": [10.0, 0.0, 10.0, 0.0],
                "fold": [1, 1, 2, 2],
                "ap": [0.2, 0.1, 0.4, 0.3],
                "train_ap": [0.5, 0.8, 0.7, 1.0],
                "auroc": [0.6, 0.5, 0.8, 0.7],
                "brier": [0.15, 0.3, 0.25, 0.5],
            },
            index=[40, 10, 30, 20],
        )
        original = results.copy(deep=True)

        summary = tune_lightgbm.summarize_search(results)

        expected = pd.DataFrame(
            {
                "configuration_id": [2, 1],
                "num_leaves": [31, 15],
                "min_child_samples": [100, 20],
                "reg_lambda": [10.0, 0.0],
                "mean_ap": [0.3, 0.2],
                "sd_ap": [sqrt(0.02), sqrt(0.02)],
                "mean_train_ap": [0.6, 0.9],
                "mean_auroc": [0.7, 0.6],
                "mean_brier": [0.2, 0.4],
            }
        )
        pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(results, original)

    def test_all_configurations_keep_fold_scores_and_reach_fresh_builders(self):
        train = pd.DataFrame({"patient_nbr": ["SYNTHETIC_A", "SYNTHETIC_B"]})
        original_train = train.copy(deep=True)
        configurations = [
            {"num_leaves": 15, "min_child_samples": 20, "reg_lambda": 0.0},
            {"num_leaves": 31, "min_child_samples": 100, "reg_lambda": 10.0},
        ]
        original_configurations = deepcopy(configurations)
        synthetic_scores = [
            pd.DataFrame({"fold": [1, 2, 3, 4, 5], "ap": [0.1] * 5}),
            pd.DataFrame({"fold": [1, 2, 3, 4, 5], "ap": [0.2] * 5}),
        ]

        with patch(
            "scripts.tune_lightgbm.evaluate_model_cv",
            side_effect=[table.copy() for table in synthetic_scores],
        ) as evaluate:
            result = tune_lightgbm.run_search(train, configurations)

        self.assertEqual(evaluate.call_count, 2)
        builders = []
        expected_tables = []
        for configuration_id, (call, configuration, scores) in enumerate(
            zip(evaluate.call_args_list, configurations, synthetic_scores), start=1
        ):
            self.assertIs(call.args[0], train)
            self.assertEqual(call.kwargs["feature_set"], "extended")
            builder = call.kwargs["model_builder"]
            builders.append(builder)
            model = builder(feature_set="extended")
            params = model.named_steps["classifier"].get_params()
            for name, value in configuration.items():
                self.assertEqual(params[name], value)
            expected = scores.assign(configuration_id=configuration_id, **configuration)
            expected_tables.append(expected)

        self.assertIsNot(builders[0], builders[1])
        pd.testing.assert_frame_equal(
            result, pd.concat(expected_tables, ignore_index=True)
        )
        pd.testing.assert_frame_equal(train, original_train)
        self.assertEqual(configurations, original_configurations)
