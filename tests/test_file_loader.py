import tempfile
import unittest
from pathlib import Path

from src.file_loader import load_data_file


CONFIG = {
    "column_aliases": {"TimeStamp": ["time"], "RecNum": ["record"], "Pressure_1": ["p1"]},
    "validation": {"units_row_min_text_fraction": 0.5},
}


class FileLoaderTests(unittest.TestCase):
    def test_csv_loading_and_alias_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("time,record,p1\n2026-01-01,1,2.5\n", encoding="utf-8")
            result = load_data_file(path, CONFIG)
        self.assertIn("TimeStamp", result.data)
        self.assertIn("RecNum", result.data)
        self.assertIn("Pressure_1", result.data)
        self.assertFalse(result.units_row_detected)


if __name__ == "__main__":
    unittest.main()

