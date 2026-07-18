import unittest

import pandas as pd
import numpy as np

from src.data_validation import calculate_sampling_intervals, parse_timestamps, validate_data


CONFIG = {
    "required_columns": ["TimeStamp"],
    "numeric_columns": ["RecNum", "Pressure_1"],
    "validation": {"timing_gap_factor": 3.0, "sampling_jitter_relative_threshold": 0.1},
}


class DataValidationTests(unittest.TestCase):
    def test_real_timestamp_is_preferred(self):
        frame=pd.DataFrame({"TimeStamp":pd.date_range("2026-01-01",periods=3,freq="100ms"),"Elapsed_Time_s":[0,9,18],"RecNum":[1,2,3]})
        result=validate_data(frame,CONFIG,.5)
        self.assertEqual(result.audit["time_source"],"recorded_timestamp")
        self.assertAlmostEqual(result.data.Elapsed_Time_s.iloc[-1],.2)

    def test_numeric_elapsed_time_works(self):
        result=validate_data(pd.DataFrame({"Elapsed_Time_s":[2.,2.1,2.3]}),CONFIG)
        self.assertEqual(result.audit["time_source"],"recorded_elapsed_time")
        self.assertEqual(result.audit["timestamp_start"],None)

    def test_record_reconstruction_gaps_and_metadata(self):
        frame=pd.DataFrame({"RecNum":[10,11,14],"Encoder":[1,2,3]})
        result=validate_data(frame,CONFIG,.25)
        self.assertEqual(result.data.Elapsed_Time_s.tolist(),[0,.25,1.])
        self.assertEqual(result.audit["record_gap_count"],1)
        self.assertTrue(result.data.Time_Reconstructed_Flag.all())
        self.assertNotIn("TimeStamp",result.data)
        self.assertFalse(result.audit["sampling_jitter_available"])
        self.assertTrue(np.isnan(result.audit["sampling_interval_cv_ratio"]))

    def test_cli_interval_overrides_config(self):
        config={**CONFIG,"intake":{"fallback_sample_interval_s":9.0}}
        result=validate_data(pd.DataFrame({"RecNum":[1,2]}),config,.2)
        self.assertAlmostEqual(result.audit["duration_s"],.2)

    def test_missing_interval_fails_clearly(self):
        with self.assertRaisesRegex(ValueError,"Supply --sample-interval-s"):
            validate_data(pd.DataFrame({"RecNum":[1,2]}),CONFIG)

    def test_invalid_record_numbers_fail(self):
        cases=[([1,1],"duplicate"),([2,1],"resets or reverses"),([1,None],"no missing")]
        for values,message in cases:
            with self.subTest(values=values),self.assertRaisesRegex(ValueError,message):
                validate_data(pd.DataFrame({"RecNum":values}),CONFIG,.1)

    def test_missing_record_number_fails(self):
        with self.assertRaisesRegex(ValueError,"RecNum is required"):
            validate_data(pd.DataFrame({"Encoder":[1,2]}),CONFIG,.1)

    def test_raw_values_and_rows_are_preserved(self):
        frame=pd.DataFrame({"RecNum":[4,5,6],"Pressure_1":[1.2,3.4,5.6]})
        result=validate_data(frame,CONFIG,.1)
        self.assertEqual(len(result.data),len(frame))
        self.assertEqual(result.data.Pressure_1.tolist(),frame.Pressure_1.tolist())

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

    def test_sampling_rate_and_jitter_reporting(self):
        frame = pd.DataFrame({
            "Original_Row_Order": range(4),
            "TimeStamp": pd.to_datetime(["2026-01-01 00:00:00.0", "2026-01-01 00:00:00.1", "2026-01-01 00:00:00.2", "2026-01-01 00:00:00.4"], format="mixed"),
            "RecNum": range(4),
            "Pressure_1": [1, 2, 3, 4],
        })
        audit = validate_data(frame, CONFIG).audit
        self.assertAlmostEqual(audit["median_derived_sampling_rate_hz"], 10.0)
        self.assertAlmostEqual(audit["effective_mean_sampling_rate_hz"], 7.5)
        self.assertAlmostEqual(audit["sampling_interval_cv_percent"], audit["sampling_interval_cv_ratio"] * 100)

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
