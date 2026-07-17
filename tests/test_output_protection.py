import tempfile
import unittest
from pathlib import Path

from src.utilities import prepare_output_directory


class OutputProtectionTests(unittest.TestCase):
    def test_nonempty_output_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing"
            target.mkdir()
            (target / "result.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(target)


if __name__ == "__main__":
    unittest.main()

