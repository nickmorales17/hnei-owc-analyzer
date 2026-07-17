import unittest

import numpy as np
import pandas as pd

from src.cycle_analysis import analyze_encoder_cycles, autocorrelation_period, classify_encoder_behavior, fft_period, flag_interval_outliers, zero_crossing_period
from src.signal_processing import process_encoder, validated_odd_window


DT = 0.02
CONFIG = {
    "filtering": {"enabled": True, "median_window_s": 0.06, "savgol_window_s": 0.5, "savgol_cycle_fraction": 0.12, "savgol_min_window_s": 0.15, "savgol_max_window_s": 1.0, "savgol_polynomial_order": 3, "detrend_enabled": False, "spike_mad_threshold": 6},
    "encoder": {"peak_prominence": 5, "trough_prominence": 5, "minimum_peak_spacing_fraction": 0.6, "zero_crossing_direction": "rising", "autocorrelation_min_period_s": 1.5, "autocorrelation_max_period_s": 10, "fft_min_period_s": 1.5, "fft_max_period_s": 10, "cycle_interval_deviation_tolerance": 0.10, "minimum_valid_intervals": 4, "method_agreement_tolerance_fraction": 0.05, "timing_confidence_threshold": 0.6},
}


def periodic_frame(period=6.0, cycles=7, spikes=True):
    time = np.arange(0, period * cycles, DT); signal = 20*np.sin(2*np.pi*time/period)
    if spikes:
        signal[np.arange(100, len(signal), 333)] += 80
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(time, unit="s")
    return pd.DataFrame({"TimeStamp": timestamps, "Encoder": signal})


class Stage4CycleTests(unittest.TestCase):
    def test_window_conversion_is_odd(self):
        self.assertEqual(validated_odd_window(0.06, 0.012, 100), 5)

    def test_median_filter_suppresses_narrow_spike(self):
        values = pd.Series([0, 0, 0, 100, 0, 0, 0], dtype=float)
        result = process_encoder(values, 0.02, CONFIG)
        self.assertLess(result.median_filtered[3], 1)
        self.assertTrue(result.spike_flags[3])

    def test_peak_trough_and_method_selection(self):
        frame = periodic_frame(); processing = process_encoder(frame.Encoder, DT, CONFIG)
        cycles, methods, summary, _ = analyze_encoder_cycles("run", frame, processing, DT, 6, CONFIG)
        peak = methods.loc[methods.method == "peak_to_peak", "period_s"].iloc[0]
        trough = methods.loc[methods.method == "trough_to_trough", "period_s"].iloc[0]
        self.assertAlmostEqual(peak, 6, delta=0.08); self.assertAlmostEqual(trough, 6, delta=0.08)
        self.assertAlmostEqual(summary["Final_Selected_Period_s"], 6, delta=0.08)
        self.assertNotEqual(summary["Selected_Timing_Method"], "unresolved")

    def test_supported_decimal_and_legacy_periods(self):
        for period in [2.0, 2.5, 3.0, 5.0, 8.0]:
            frame = periodic_frame(period=period, cycles=8, spikes=False)
            processing = process_encoder(frame.Encoder, DT, CONFIG, period)
            _, _, summary, _ = analyze_encoder_cycles(f"run_{period}", frame, processing, DT, float(period), CONFIG)
            self.assertAlmostEqual(summary["Final_Selected_Period_s"], period, delta=0.08)

    def test_autocorrelation_fft_and_zero_crossing(self):
        frame = periodic_frame(spikes=False); values = frame.Encoder.to_numpy()
        self.assertAlmostEqual(autocorrelation_period(values, DT, 4, 9)["period_s"], 6, delta=0.08)
        self.assertAlmostEqual(fft_period(values, DT, 4, 9)["period_s"], 6, delta=0.08)
        self.assertAlmostEqual(zero_crossing_period(values, DT)["period_s"], 6, delta=0.08)

    def test_interval_outlier_flagging(self):
        flags = flag_interval_outliers(np.array([6.0, 6.02, 6.0, 7.0]), 0.10)
        self.assertEqual(flags.tolist(), [False, False, False, True])

    def test_low_cycle_count_warning(self):
        frame = periodic_frame(cycles=3, spikes=False); processing = process_encoder(frame.Encoder, DT, CONFIG)
        _, _, summary, _ = analyze_encoder_cycles("short", frame, processing, DT, 6, CONFIG)
        self.assertIn("Fewer than 4", summary["Low_Cycle_Count_Warning"])

    def test_explicit_all_valid_cv_and_required_summary_fields(self):
        frame = periodic_frame(spikes=False); processing = process_encoder(frame.Encoder, DT, CONFIG, 6)
        _, _, summary, _ = analyze_encoder_cycles("run", frame, processing, DT, 6, CONFIG)
        required = {"All_Interval_Count","Valid_Interval_Count","All_Interval_Mean_s","Valid_Interval_Mean_s","All_Interval_Median_s","Valid_Interval_Median_s","All_Population_Std_Cycle_s","All_Sample_Std_Cycle_s","Valid_Population_Std_Cycle_s","Valid_Sample_Std_Cycle_s","All_Population_CV_Ratio","All_Population_CV_Percent","All_Sample_CV_Ratio","All_Sample_CV_Percent","Valid_Population_CV_Ratio","Valid_Population_CV_Percent","Valid_Sample_CV_Ratio","Valid_Sample_CV_Percent","Valid_RMS_Cycle_s","Valid_Mean_Measured_Frequency_Hz","Valid_Sample_Frequency_Std_Hz","Valid_Minimum_Cycle_s","Valid_Maximum_Cycle_s","Valid_Peak_To_Peak_Range_s","Low_Cycle_Count_Warning","Final_Selected_Period_s","Final_Selected_Frequency_Hz","Selected_Timing_Method","Selection_Reason","Method_Agreement_Fraction","Timing_Method_Confidence"}
        self.assertTrue(required.issubset(summary))
        self.assertAlmostEqual(summary["Valid_Population_CV_Percent"], summary["Valid_Population_CV_Ratio"]*100)
        self.assertAlmostEqual(summary["Valid_Sample_CV_Percent"], summary["Valid_Sample_CV_Ratio"]*100)

    def test_population_and_sample_variability_use_matching_ddof(self):
        frame = periodic_frame(period=6, cycles=7, spikes=False)
        # Add deterministic timing modulation so interval variability is nonzero.
        frame["Encoder"] = 20*np.sin(2*np.pi*(np.arange(len(frame))*DT + 0.03*np.sin(np.arange(len(frame))*DT/4))/6)
        processing = process_encoder(frame.Encoder, DT, CONFIG, 6)
        cycles, _, summary, _ = analyze_encoder_cycles("run", frame, processing, DT, 6, CONFIG)
        values = cycles.loc[~cycles.Interval_Outlier_Flag, "Measured_Cycle_s"].to_numpy()
        self.assertGreaterEqual(len(values), 2)
        self.assertAlmostEqual(summary["Valid_Population_Std_Cycle_s"], np.std(values, ddof=0))
        self.assertAlmostEqual(summary["Valid_Sample_Std_Cycle_s"], np.std(values, ddof=1))
        self.assertNotEqual(summary["Valid_Population_Std_Cycle_s"], summary["Valid_Sample_Std_Cycle_s"])
        self.assertAlmostEqual(summary["Valid_Population_CV_Ratio"], np.std(values, ddof=0)/np.mean(values))
        self.assertAlmostEqual(summary["Valid_Sample_CV_Ratio"], np.std(values, ddof=1)/np.mean(values))

    def test_final_selection_does_not_overwrite_interval_mean(self):
        frame = periodic_frame(); processing = process_encoder(frame.Encoder, DT, CONFIG, 6)
        cycles, _, summary, _ = analyze_encoder_cycles("run", frame, processing, DT, 6, CONFIG)
        valid_mean = cycles.loc[~cycles.Interval_Outlier_Flag, "Measured_Cycle_s"].mean()
        self.assertAlmostEqual(summary["Valid_Interval_Mean_s"], valid_mean)
        self.assertIn("Final_Selected_Period_s", summary)
        self.assertIn("Selection_Reason", summary)

    def test_single_interval_sample_variability_is_unavailable(self):
        frame = periodic_frame(period=6, cycles=2.7, spikes=False); processing = process_encoder(frame.Encoder, DT, CONFIG, 6)
        _, _, summary, _ = analyze_encoder_cycles("one", frame, processing, DT, 6, CONFIG)
        if summary["Valid_Interval_Count"] == 1:
            self.assertTrue(np.isnan(summary["Valid_Sample_Std_Cycle_s"]))
            self.assertTrue(np.isnan(summary["Valid_Sample_CV_Ratio"]))

    def test_encoder_behavior_is_processed_periodic(self):
        frame = periodic_frame(); processing = process_encoder(frame.Encoder, DT, CONFIG)
        behavior = classify_encoder_behavior(frame.Encoder, processing.smoothed, DT)
        self.assertEqual(behavior["encoder_behavior"], "processed periodic position")
        self.assertIn("does not require counts per revolution", behavior["classification_basis"])


if __name__ == "__main__":
    unittest.main()
