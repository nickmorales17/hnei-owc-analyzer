import unittest

import pandas as pd

from src.data_validation import calculate_sampling_intervals, parse_timestamps, validate_data


CONFIG = {
    "required_columns": ["TimeStamp"],
    "numeric_columns": ["RecNum", "Pressure_1"],
    "validation": {"timing_gap_factor": 3.0, "sampling_jitter_relative_threshold": 0.1},
}


class DataValidationTests(unittest.TestCase):
    def test_excel_serial_timestamp_conversion(self):
        parsed = parse_timestamps(pd.Series([46217.0, 46217.5]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-07-14 00:00:00"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2026-07-14 12:00:00"))

    def test_sampling_interval_calculation(self):
        timestamps = pd.Series(
            pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:00:00.1", "2026-01-01 00:00:00.2"],
                format="mixed",
            )
        )
        intervals = calculate_sampling_intervals(timestamps)
        self.assertAlmostEqual(intervals.iloc[1], 0.1)
        self.assertAlmostEqual(intervals.iloc[2], 0.1)

    def test_duplicate_and_record_gap_detection(self):
        frame = pd.DataFrame({
            "Original_Row_Order": [0, 1, 2, 3],
            "TimeStamp": [46217.0, 46217.0, 46217.000001, 46217.000002],
            "RecNum": [1, 1, 2, 4],
            "Pressure_1": [10, 10, 11, 12],
        })
        result = validate_data(frame, CONFIG)
        self.assertEqual(result.audit["duplicate_timestamp_count"], 2)
        self.assertEqual(result.audit["duplicate_record_number_count"], 2)
        self.assertEqual(result.audit["record_gap_count"], 1)

    def test_numeric_conversion_failures_are_flagged(self):
        frame = pd.DataFrame({"Original_Row_Order": [0], "TimeStamp": [46217.0], "RecNum": [1], "Pressure_1": ["bad"]})
        result = validate_data(frame, CONFIG)
        self.assertEqual(result.audit["numeric_conversion_failures:Pressure_1"], 1)
        self.assertIn("numeric_conversion_failure:Pressure_1", result.data.loc[0, "Quality_Flags"])


if __name__ == "__main__":
    unittest.main()
