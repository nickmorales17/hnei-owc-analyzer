import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.steady_state import classify_steady_state
from src.test_segmentation import RunBlock, detect_active_blocks, estimate_preliminary_period, group_recorded_targets, match_expected_period


DT = 0.05
CONFIG = {
    "expected_cycle_times_s": [5, 6, 7, 8],
    "target_cycle_tolerance_s": 0.1,
    "inferred_label_confidence_threshold": 0.6,
    "activity_detection": {
        "signals": ["Encoder", "Pressure_1", "Pressure_2", "Torque"], "rolling_window_s": 0.5,
        "encoder_std_threshold": 0.5, "encoder_range_threshold": 1.0, "supporting_noise_multiplier": 4,
        "active_score_threshold": 1, "minimum_active_duration_s": 4, "minimum_idle_duration_s": 1,
        "block_merge_gap_s": 0.4, "edge_cleanup_s": 0.1,
    },
    "period_estimation": {
        "smoothing_window_s": 0.5, "peak_prominence": 5, "minimum_peak_spacing_s": 3.5,
        "autocorrelation_min_period_s": 4, "autocorrelation_max_period_s": 9, "expected_match_tolerance_s": 0.8,
    },
    "steady_state": {"minimum_steady_cycles": 3, "period_stability_tolerance": 0.12, "amplitude_stability_tolerance": 0.35},
    "manual_run_boundary_overrides": [], "manual_steady_state_overrides": [],
}


def synthetic_four_blocks():
    rng = np.random.default_rng(1234)
    values = []
    pressure = []
    blocks = []
    cursor = 0
    for period in [5, 6, 7, 8]:
        idle_n = int(2 / DT)
        values.extend(rng.normal(0, 0.05, idle_n)); pressure.extend(rng.normal(0, 0.03, idle_n)); cursor += idle_n
        n = int((period * 6) / DT); local = np.arange(n) * DT
        envelope = np.ones(n); ramp = int(period / DT); envelope[:ramp] = np.linspace(0.2, 1, ramp); envelope[-ramp:] = np.linspace(1, 0.1, ramp)
        encoder = 20 * envelope * np.sin(2 * np.pi * local / period) + rng.normal(0, 0.15, n)
        p = 8 * envelope * np.sin(2 * np.pi * (local - 0.3) / period) + rng.normal(0, 0.1, n)
        start = cursor; values.extend(encoder); pressure.extend(p); cursor += n; blocks.append((start, cursor - 1))
    idle_n = int(2 / DT); values.extend(rng.normal(0, 0.05, idle_n)); pressure.extend(rng.normal(0, 0.03, idle_n))
    count = len(values); timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(count) * DT, unit="s")
    data = pd.DataFrame({"TimeStamp": timestamps, "Encoder": values, "Pressure_1": pressure, "Pressure_2": pressure, "Torque": np.asarray(pressure) * 0.4})
    return data, blocks


class Stage3SegmentationTests(unittest.TestCase):
    def test_contiguous_and_repeated_recorded_targets(self):
        data = pd.DataFrame({"Target_Cycle_s": [5, 5.02, np.nan, 5, 5, 6, 6]})
        blocks = group_recorded_targets(data, 0.1)
        self.assertEqual([(b.start_row, b.end_row) for b in blocks], [(0, 1), (3, 4), (5, 6)])
        self.assertEqual([b.provisional_target_cycle_s for b in blocks], [5, 5, 6])

    def test_decimal_recorded_target_remains_float(self):
        data = pd.DataFrame({"Target_Cycle_s": [2.5, 2.5, np.nan, 3.0]})
        blocks = group_recorded_targets(data, 0.01)
        self.assertEqual(blocks[0].provisional_target_cycle_s, 2.5)
        self.assertIsInstance(blocks[0].provisional_target_cycle_s, float)

    def test_active_blocks_idle_gaps_and_four_periods(self):
        data, _ = synthetic_four_blocks()
        blocks = detect_active_blocks(data, DT, CONFIG)
        self.assertEqual(len(blocks), 4)
        estimates = [estimate_preliminary_period(data.loc[b.start_row:b.end_row, "Encoder"], DT, CONFIG)["selected_period_s"] for b in blocks]
        np.testing.assert_allclose(estimates, [5, 6, 7, 8], atol=0.35)

    def test_short_gap_is_merged(self):
        data, _ = synthetic_four_blocks()
        active = data["Encoder"].abs() > 1
        middle = active[active].index[len(active[active]) // 8]
        data.loc[middle:middle + int(0.2 / DT), ["Encoder", "Pressure_1", "Pressure_2", "Torque"]] = 0
        self.assertEqual(len(detect_active_blocks(data, DT, CONFIG)), 4)

    def test_peak_and_autocorrelation_estimates(self):
        time = np.arange(0, 60, DT)
        estimate = estimate_preliminary_period(pd.Series(20 * np.sin(2 * np.pi * time / 6)), DT, CONFIG)
        self.assertAlmostEqual(estimate["peak_period_s"], 6, delta=0.1)
        self.assertAlmostEqual(estimate["autocorrelation_period_s"], 6, delta=0.1)

    def test_expected_match_and_low_confidence(self):
        label, confidence = match_expected_period(7.05, 0.95, CONFIG)
        self.assertEqual(label, 7)
        label, confidence = match_expected_period(6.5, 0.0, CONFIG)
        self.assertIsNone(label)

    def test_decimal_inferred_target_is_not_rounded(self):
        config = {**CONFIG, "expected_cycle_times_s": [2.0, 2.5, 3.0], "period_estimation": {**CONFIG["period_estimation"], "expected_match_tolerance_s": 0.35}}
        label, confidence = match_expected_period(2.48, 0.95, config)
        self.assertEqual(label, 2.5)
        self.assertIsInstance(label, float)

    def test_manual_run_override(self):
        data, _ = synthetic_four_blocks(); config = {**CONFIG, "manual_run_boundary_overrides": [{"run_id": "manual_1", "start_row": 10, "end_row": 100}]}
        blocks = detect_active_blocks(data, DT, config)
        self.assertEqual((blocks[0].start_row, blocks[0].end_row, blocks[0].target_source), (10, 100, "manual"))

    def test_file_beginning_partway_through_active_run_starts_at_zero(self):
        time = np.arange(0, 30, DT)
        data = pd.DataFrame({
            "TimeStamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(time, unit="s"),
            "Encoder": 20*np.sin(2*np.pi*(time+1.7)/5),
            "Pressure_1": 8*np.sin(2*np.pi*(time+1.4)/5),
            "Pressure_2": 8*np.sin(2*np.pi*(time+1.4)/5),
            "Torque": 2*np.sin(2*np.pi*(time+1.4)/5),
        })
        blocks = detect_active_blocks(data, DT, CONFIG)
        self.assertEqual(blocks[0].start_row, 0)

    def test_no_sample_specific_boundary_constants_in_source(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src").glob("*.py"))
        for boundary in [2116, 2335, 5511, 5669, 8731, 8957, 12390, 12729]:
            self.assertNotIn(str(boundary), source)

    def test_startup_stopping_and_missing_startup(self):
        data, true_blocks = synthetic_four_blocks(); block = RunBlock("run_test", *true_blocks[1], "inferred")
        cycles, summary, _ = classify_steady_state(data, block, DT, CONFIG)
        self.assertGreaterEqual(summary["steady_cycle_count"], 3)
        steady = cycles[cycles["is_steady_state"]]
        self.assertGreater(steady.start_row.min(), block.start_row)
        self.assertLess(steady.end_row.max(), block.end_row)
        first = RunBlock("first", 0, true_blocks[0][1] - true_blocks[0][0], "inferred")
        captured_start = true_blocks[0][0] + int(5 / DT)
        active_from_start = data.loc[captured_start:true_blocks[0][1]].reset_index(drop=True)
        active_from_start["TimeStamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(len(active_from_start)) * DT, unit="s")
        _, first_summary, _ = classify_steady_state(active_from_start, first, DT, CONFIG)
        self.assertTrue(first_summary["startup_not_captured"])

    def test_manual_steady_override(self):
        data, blocks = synthetic_four_blocks(); block = RunBlock("run_x", *blocks[0], "inferred")
        config = {**CONFIG, "manual_steady_state_overrides": [{"run_id": "run_x", "start_row": block.start_row + 10, "end_row": block.end_row - 10}]}
        _, summary, _ = classify_steady_state(data, block, DT, config)
        self.assertEqual(summary["selection_source"], "manual")


if __name__ == "__main__":
    unittest.main()
