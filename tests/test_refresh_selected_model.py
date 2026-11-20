from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.refresh_selected_model import main


class RefreshSelectedModelGuardTest(unittest.TestCase):
    def test_existing_destination_is_refused_before_loading_or_fitting(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.refresh_selected_model.load_verified_partitions") as load:
                with self.assertRaises(FileExistsError):
                    main(Path(directory))
                load.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_changed_cv_evidence_is_rejected_before_loading_or_fitting(self):
        with TemporaryDirectory() as directory:
            with patch("scripts.refresh_selected_model.load_verified_partitions") as load, patch(
                "scripts.refresh_selected_model.file_sha256", return_value="changed-hash"
            ):
                load.side_effect = AssertionError("Training must not start with changed CV evidence")
                with self.assertRaisesRegex(ValueError, "CV evidence"):
                    main(Path(directory) / "new")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
