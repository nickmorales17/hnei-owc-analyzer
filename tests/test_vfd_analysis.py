import unittest

import numpy as np
import pandas as pd

from src.vfd_analysis import calculate_vfd_command, cap_command_mv, interpret_command_comparison, verify_vfd_run


CONFIG = {"vfd_command": {"cycle_frequency_numerator": 120, "command_frequency_offset_hz": 1.14, "command_slope_hz_per_mv": 0.00626, "command_min_mv": 0, "command_max_mv": 4000}}


class VFDAnalysisTests(unittest.TestCase):
    def test_expected_nominal_commands(self):
        expected = {8: 2578.274760, 7: 2920.584208, 6: 3376.996805}
        for cycle, command in expected.items():
            result = calculate_vfd_command(cycle, CONFIG)
            self.assertAlmostEqual(result["Desired_Target_Frequency_Hz"], 120/cycle, places=9)
            self.assertAlmostEqual(result["Reconstructed_Command_mV_Uncapped"], command, places=5)
            self.assertAlmostEqual(result["Command_Equivalent_Cycle_s"], cycle, places=9)

    def test_upper_cap_and_equivalent_cycle(self):
        result = calculate_vfd_command(5, CONFIG)
        self.assertAlmostEqual(result["Reconstructed_Command_mV_Uncapped"], 4015.974441, places=5)
        self.assertEqual(result["Reconstructed_Command_mV_Capped"], 4000)
        self.assertAlmostEqual(result["Command_Equivalent_Frequency_Hz"], 23.9, places=9)
        self.assertAlmostEqual(result["Command_Equivalent_Cycle_s"], 5.020920502, places=8)
        self.assertTrue(result["Command_Saturation_Flag"])

    def test_saturation_is_generic_for_short_targets(self):
        for cycle in [2.0, 2.5, 3.0, 5.0]:
            self.assertTrue(calculate_vfd_command(cycle, CONFIG)["Command_Saturation_Flag"])
        self.assertFalse(calculate_vfd_command(8.0, CONFIG)["Command_Saturation_Flag"])

    def test_lower_cap(self):
        self.assertEqual(cap_command_mv(-10, 0, 4000), 0)

    def test_recorded_discrepancies_and_cycle_errors(self):
        data = pd.DataFrame({"Target_VFD_Hz": [24,24], "VFD_Command_mV": [3990,3990]})
        result = verify_vfd_run("run", data, 5, "recorded", 5.01, CONFIG)
        self.assertAlmostEqual(result["Frequency_Discrepancy_Hz"], 0)
        self.assertAlmostEqual(result["Command_Discrepancy_mV"], -10)
        self.assertAlmostEqual(result["Signed_Error_From_Nominal_s"], 0.01)
        self.assertAlmostEqual(result["Error_From_Capped_Command_s"], 5.01-120/23.9)
        self.assertAlmostEqual(result["Final_Measured_Cycle_Frequency_Hz"], 1/5.01)
        self.assertAlmostEqual(result["Final_Measured_VFD_Equivalent_Frequency_Hz"], 120/5.01)
        self.assertEqual(result["Frequency_Source"], "recorded")

    def test_vfd_errors_use_supplied_final_selected_period(self):
        final_selected_period = 5.01
        unrelated_interval_mean = 5.20
        result = verify_vfd_run("run", pd.DataFrame(), 5, "inferred", final_selected_period, CONFIG)
        self.assertAlmostEqual(result["Final_Measured_Cycle_s"], final_selected_period)
        self.assertAlmostEqual(result["Signed_Error_From_Nominal_s"], final_selected_period-5)
        self.assertNotAlmostEqual(result["Signed_Error_From_Nominal_s"], unrelated_interval_mean-5)

    def test_missing_recorded_values_are_reconstructed(self):
        result = verify_vfd_run("run", pd.DataFrame({"Encoder":[1]}), 6, "inferred", 6.01, CONFIG)
        self.assertEqual(result["Frequency_Source"], "reconstructed")
        self.assertEqual(result["Command_Source"], "reconstructed")
        self.assertTrue(np.isnan(result["Recorded_Command_mV"]))

    def test_unverified_new_turbine_scaling_is_unavailable(self):
        unavailable = {"vfd_command": {"enabled": False}}
        result = calculate_vfd_command(2.5, unavailable)
        self.assertEqual(result["VFD_Verification_Status"], "unavailable_unverified_scaling")
        self.assertTrue(np.isnan(result["Reconstructed_Command_mV_Capped"]))

    def test_five_second_interpretation_is_not_distinguishable(self):
        result = interpret_command_comparison(5.0, 5.020920502, 5.000951, 0.012, 2, 0.036)
        self.assertFalse(result["Can_Distinguish_Nominal_From_Capped"])
        self.assertIn("numerically closer to the nominal", result["Command_Interpretation"])
        self.assertIn("do not reliably distinguish", result["Command_Interpretation"])

    def test_unsaturated_identical_expectations_need_no_distinction(self):
        result = interpret_command_comparison(6.0, 6.0, 6.01, 0.012, 4, 0.04)
        self.assertIn("identical", result["Command_Interpretation"])


if __name__ == "__main__":
    unittest.main()
