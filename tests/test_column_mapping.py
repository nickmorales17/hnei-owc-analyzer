import unittest

from src.column_mapping import map_columns


class ColumnMappingTests(unittest.TestCase):
    def test_common_aliases_map_to_canonical_names(self):
        aliases = {"TimeStamp": ["time"], "Pressure_1": ["p1"]}
        result = map_columns(["time", "p1"], aliases)
        self.assertEqual(result.rename_map, {"time": "TimeStamp", "p1": "Pressure_1"})

    def test_ambiguous_alias_is_not_guessed(self):
        aliases = {"Pressure_1": ["pressure"], "Pressure_2": ["pressure"]}
        result = map_columns(["pressure"], aliases)
        self.assertEqual(result.rename_map, {})
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()

