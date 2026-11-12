"""Guard the frozen evaluation's I/O boundary without fitting real models."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.evaluate_final_models import main


class FinalEvaluationGuardTest(unittest.TestCase):
    def test_existing_directory_is_refused_before_loading_data(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.evaluate_final_models.load_verified_partitions") as load:
                with self.assertRaises(FileExistsError):
                    main(Path(directory))
                load.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_bootstrap_count_is_rejected_without_creating_output(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "not_created"
            with patch("scripts.evaluate_final_models.load_verified_partitions") as load:
                with self.assertRaises(ValueError):
                    main(destination, n_bootstrap=0)
                load.assert_not_called()
            self.assertFalse(destination.exists())
