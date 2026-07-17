"""Reconstructed and recorded VFD command verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cap_command_mv(command_mv: float, minimum_mv: float, maximum_mv: float) -> float:
    """Clamp a reconstructed command to configured electrical limits."""
    return min(max(float(command_mv), float(minimum_mv)), float(maximum_mv))


def calculate_vfd_command(target_cycle_s: float, config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("vfd_command", {})
    if target_cycle_s <= 0:
        raise ValueError("Target cycle time must be a positive floating-point value.")
    if not bool(settings.get("enabled", True)):
        return {"VFD_Verification_Status": "unavailable_unverified_scaling", "Desired_Target_Frequency_Hz": np.nan, "Reconstructed_Command_mV_Uncapped": np.nan, "Reconstructed_Command_mV_Capped": np.nan, "Command_Equivalent_Frequency_Hz": np.nan, "Command_Equivalent_Cycle_s": np.nan, "Command_Saturation_Flag": False}
    numerator = float(settings.get("cycle_frequency_numerator", 120.0)); offset = float(settings.get("command_frequency_offset_hz", 1.14)); slope = float(settings.get("command_slope_hz_per_mv", 0.00626)); lower = float(settings.get("command_min_mv", 0)); upper = float(settings.get("command_max_mv", 4000))
    desired = numerator / target_cycle_s; uncapped = (desired + offset) / slope; capped = cap_command_mv(uncapped, lower, upper); equivalent_frequency = slope*capped-offset; equivalent_cycle = numerator/equivalent_frequency if equivalent_frequency > 0 else np.nan
    return {"VFD_Verification_Status": "available_reconstructed", "Desired_Target_Frequency_Hz": desired, "Reconstructed_Command_mV_Uncapped": uncapped, "Reconstructed_Command_mV_Capped": capped, "Command_Equivalent_Frequency_Hz": equivalent_frequency, "Command_Equivalent_Cycle_s": equivalent_cycle, "Command_Saturation_Flag": bool(not np.isclose(uncapped, capped))}


def interpret_command_comparison(nominal_cycle_s: float, capped_cycle_s: float, measured_cycle_s: float, median_sampling_interval_s: float, valid_interval_count: int, measured_range_s: float) -> dict[str, Any]:
    """Assess distinguishability without treating numerical closeness as proof."""
    if not all(np.isfinite(value) for value in [nominal_cycle_s, capped_cycle_s, measured_cycle_s]):
        return {"Nominal_To_Capped_Difference_s": np.nan, "Can_Distinguish_Nominal_From_Capped": False, "Command_Interpretation": "VFD command comparison is unavailable."}
    expectation_difference = abs(capped_cycle_s - nominal_cycle_s)
    if np.isclose(expectation_difference, 0.0):
        return {"Nominal_To_Capped_Difference_s": 0.0, "Can_Distinguish_Nominal_From_Capped": False, "Command_Interpretation": "Nominal and capped-command expectations are identical; no saturation distinction is required."}
    closer = "nominal" if abs(measured_cycle_s-nominal_cycle_s) <= abs(measured_cycle_s-capped_cycle_s) else "capped"
    resolution_limited = median_sampling_interval_s >= expectation_difference/3
    variation_similar = np.isfinite(measured_range_s) and measured_range_s >= expectation_difference/2
    low_count = valid_interval_count < 4
    distinguishable = not (resolution_limited or variation_similar or low_count)
    if distinguishable:
        wording = f"Measured timing is closer to the {closer} expectation and the configured resolution/event checks support distinguishing the expectations; this is not proof of causation."
    else:
        wording = f"Measured timing is numerically closer to the {closer} expectation, but the nominal-to-capped difference ({expectation_difference:.6f} s), median sampling interval ({median_sampling_interval_s:.3f} s), available interval count ({valid_interval_count}), and measured peak-to-peak variation ({measured_range_s:.3f} s) do not reliably distinguish nominal from capped behavior. Recorded CR1000X command voltage or measured VFD output frequency is needed for confirmation."
    return {"Nominal_To_Capped_Difference_s": expectation_difference, "Can_Distinguish_Nominal_From_Capped": distinguishable, "Command_Interpretation": wording}


def verify_vfd_run(run_id: str, run_data: pd.DataFrame, target_cycle_s: float, target_source: str, measured_cycle_s: float, config: dict[str, Any], median_sampling_interval_s: float = np.nan, valid_interval_count: int = 0, measured_range_s: float = np.nan) -> dict[str, Any]:
    result: dict[str, Any] = {"Run_ID": run_id, "Nominal_Target_Cycle_s": target_cycle_s, "Target_Source": target_source, **calculate_vfd_command(target_cycle_s, config)}
    recorded_frequency = float(run_data["Target_VFD_Hz"].dropna().median()) if "Target_VFD_Hz" in run_data and run_data["Target_VFD_Hz"].notna().any() else np.nan
    recorded_command = float(run_data["VFD_Command_mV"].dropna().median()) if "VFD_Command_mV" in run_data and run_data["VFD_Command_mV"].notna().any() else np.nan
    numerator_value = config.get("vfd_command", {}).get("cycle_frequency_numerator", 120.0)
    numerator = float(numerator_value) if numerator_value is not None else np.nan
    result.update({"Recorded_Target_Frequency_Hz": recorded_frequency, "Recorded_Command_mV": recorded_command, "Frequency_Source": "recorded" if np.isfinite(recorded_frequency) else "reconstructed", "Command_Source": "recorded" if np.isfinite(recorded_command) else "reconstructed", "Frequency_Discrepancy_Hz": recorded_frequency-result["Desired_Target_Frequency_Hz"] if np.isfinite(recorded_frequency) else np.nan, "Command_Discrepancy_mV": recorded_command-result["Reconstructed_Command_mV_Capped"] if np.isfinite(recorded_command) else np.nan, "Final_Measured_Cycle_s": measured_cycle_s, "Final_Measured_Cycle_Frequency_Hz": 1/measured_cycle_s if np.isfinite(measured_cycle_s) and measured_cycle_s>0 else np.nan, "Final_Measured_VFD_Equivalent_Frequency_Hz": numerator/measured_cycle_s if np.isfinite(measured_cycle_s) and measured_cycle_s>0 else np.nan})
    nominal_error = measured_cycle_s-target_cycle_s if np.isfinite(measured_cycle_s) else np.nan; capped_error = measured_cycle_s-result["Command_Equivalent_Cycle_s"] if np.isfinite(measured_cycle_s) and np.isfinite(result["Command_Equivalent_Cycle_s"]) else np.nan
    result.update({"Signed_Error_From_Nominal_s": nominal_error, "Absolute_Error_From_Nominal_s": abs(nominal_error) if np.isfinite(nominal_error) else np.nan, "Percent_Error_From_Nominal": 100*nominal_error/target_cycle_s if np.isfinite(nominal_error) else np.nan, "Error_From_Capped_Command_s": capped_error, "Percent_Error_From_Capped_Command": 100*capped_error/result["Command_Equivalent_Cycle_s"] if np.isfinite(capped_error) else np.nan, "Behavior_Consistency": "closer_to_capped_command_expectation" if np.isfinite(capped_error) and abs(capped_error)<abs(nominal_error) else "closer_to_nominal_target" if np.isfinite(nominal_error) else "unresolved"})
    result.update(interpret_command_comparison(target_cycle_s, result["Command_Equivalent_Cycle_s"], measured_cycle_s, median_sampling_interval_s, valid_interval_count, measured_range_s))
    return result
