import unittest

from src.file_loader import load_config


class ConfigurationProfileTests(unittest.TestCase):
    def test_legacy_profile_preserves_verified_vfd_equation(self):
        config = load_config("config/legacy_config.yaml")
        self.assertEqual(config["expected_cycle_times_s"], [5.0, 6.0, 7.0, 8.0])
        self.assertTrue(config["vfd_command"]["enabled"])

    def test_new_turbine_profile_preserves_decimal_and_disables_vfd(self):
        config = load_config("config/new_turbine_config.yaml")
        self.assertEqual(config["expected_cycle_times_s"], [5.0, 4.0, 3.0, 2.5, 2.0])
        self.assertIsInstance(config["expected_cycle_times_s"][3], float)
        self.assertEqual(config["intake"]["fallback_sample_interval_s"], 0.012)
        self.assertFalse(config["vfd_command"]["enabled"])
        self.assertIsNone(config["vfd_command"]["command_slope_hz_per_mv"])


if __name__ == "__main__":
    unittest.main()
