"""Cycle-level and steady-state classification for detected runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .test_segmentation import RunBlock, _smooth, seconds_to_samples


def _longest_consecutive(values: list[int]) -> list[int]:
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return max(groups, key=len)


def classify_steady_state(
    data: pd.DataFrame, block: RunBlock, sampling_interval_s: float, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """Classify complete cycles and select the longest stable sequence."""
    settings = config.get("steady_state", {})
    overrides = config.get("manual_steady_state_overrides", []) or settings.get("manual_overrides", [])
    for item in overrides:
        if str(item.get("run_id")) == block.run_id:
            start, end = int(item["start_row"]), int(item["end_row"])
            summary = {"run_id": block.run_id, "steady_start_row": start, "steady_end_row": end, "steady_cycle_count": item.get("steady_cycle_count", np.nan), "startup_not_captured": bool(item.get("startup_not_captured", False)), "selection_source": "manual", "warning": "Manual steady-state override applied."}
            return pd.DataFrame(), summary, np.array([], dtype=int)

    segment = data.loc[block.start_row : block.end_row]
    encoder = segment["Encoder"].interpolate(limit_direction="both").to_numpy(dtype=float)
    period_settings = config.get("period_estimation", {})
    window = seconds_to_samples(float(period_settings.get("smoothing_window_s", 1.0)), sampling_interval_s, 5)
    processed = _smooth(encoder, window)
    distance = seconds_to_samples(float(period_settings.get("minimum_peak_spacing_s", 4.0)), sampling_interval_s)
    peaks, _ = find_peaks(processed, prominence=period_settings.get("peak_prominence", 60.0), distance=distance)
    global_peaks = peaks + block.start_row
    cycle_rows: list[dict[str, Any]] = []
    for number, (left, right) in enumerate(zip(global_peaks[:-1], global_peaks[1:]), 1):
        cycle = data.loc[left:right]
        row = {
            "run_id": block.run_id, "cycle_number": number, "start_row": int(left), "end_row": int(right),
            "period_s": (right - left) * sampling_interval_s,
            "encoder_amplitude": float(cycle["Encoder"].max() - cycle["Encoder"].min()),
            "pressure_1_amplitude": float(cycle["Pressure_1"].max() - cycle["Pressure_1"].min()) if "Pressure_1" in cycle else np.nan,
            "pressure_2_amplitude": float(cycle["Pressure_2"].max() - cycle["Pressure_2"].min()) if "Pressure_2" in cycle else np.nan,
            "torque_rms": float(np.sqrt(np.nanmean(np.square(cycle["Torque"])))) if "Torque" in cycle else np.nan,
            "encoder_baseline": float(cycle["Encoder"].median()),
        }
        cycle_rows.append(row)
    cycles = pd.DataFrame(cycle_rows)
    if cycles.empty:
        return cycles, {"run_id": block.run_id, "steady_start_row": np.nan, "steady_end_row": np.nan, "steady_cycle_count": 0, "startup_not_captured": block.start_row <= seconds_to_samples(1.0, sampling_interval_s), "selection_source": "automatic", "warning": "Too few complete cycles for steady-state detection."}, global_peaks

    period_tol = float(settings.get("period_stability_tolerance", 0.12))
    amp_tol = float(settings.get("amplitude_stability_tolerance", 0.35))
    period_median = cycles["period_s"].median()
    amp_median = cycles["encoder_amplitude"].median()
    period_ok = abs(cycles["period_s"] - period_median) / period_median <= period_tol
    amplitude_ok = abs(cycles["encoder_amplitude"] - amp_median) / amp_median <= amp_tol
    cycles["is_stable"] = period_ok & amplitude_ok
    cycles["exclusion_reason"] = ""
    cycles.loc[~period_ok, "exclusion_reason"] = "cycle_period_unstable"
    cycles.loc[period_ok & ~amplitude_ok, "exclusion_reason"] = "encoder_amplitude_unstable"
    stable_indexes = _longest_consecutive(cycles.index[cycles["is_stable"]].tolist())
    minimum = int(settings.get("minimum_steady_cycles", 3))
    warning = ""
    if len(stable_indexes) < minimum:
        stable_indexes = []
        warning = f"No reliable sequence of at least {minimum} steady cycles."
    cycles["is_steady_state"] = cycles.index.isin(stable_indexes)
    if stable_indexes:
        steady_start = int(cycles.loc[stable_indexes[0], "start_row"])
        steady_end = int(cycles.loc[stable_indexes[-1], "end_row"])
    else:
        steady_start = steady_end = np.nan
    startup_not_captured = block.start_row <= seconds_to_samples(1.0, sampling_interval_s) and float(np.std(encoder[:window])) >= float(config.get("activity_detection", {}).get("encoder_std_threshold", 10.0))
    summary = {"run_id": block.run_id, "steady_start_row": steady_start, "steady_end_row": steady_end, "steady_cycle_count": len(stable_indexes), "startup_not_captured": bool(startup_not_captured), "selection_source": "automatic", "warning": warning}
    return cycles, summary, global_peaks


def annotate_operating_states(data: pd.DataFrame, blocks: list[RunBlock], results: dict[str, tuple[pd.DataFrame, dict[str, Any], np.ndarray]]) -> pd.DataFrame:
    annotated = data.copy()
    annotated["Run_ID"] = ""
    annotated["Inferred_Target_Cycle_s"] = np.nan
    annotated["Target_Source"] = ""
    annotated["Operating_State"] = "idle"
    annotated["Cycle_Number"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["Is_Steady_State"] = False
    annotated["Detection_Confidence"] = np.nan
    for block in blocks:
        mask = annotated.index.to_series().between(block.start_row, block.end_row)
        annotated.loc[mask, "Run_ID"] = block.run_id
        annotated.loc[mask, "Inferred_Target_Cycle_s"] = block.provisional_target_cycle_s
        annotated.loc[mask, "Target_Source"] = block.target_source
        annotated.loc[mask, "Operating_State"] = "unclassified"
        annotated.loc[mask, "Detection_Confidence"] = block.detection_confidence
        cycles, summary, _ = results[block.run_id]
        for _, cycle in cycles.iterrows():
            cmask = annotated.index.to_series().between(int(cycle.start_row), int(cycle.end_row))
            annotated.loc[cmask, "Cycle_Number"] = int(cycle.cycle_number)
        if pd.notna(summary["steady_start_row"]):
            start, end = int(summary["steady_start_row"]), int(summary["steady_end_row"])
            annotated.loc[annotated.index.to_series().between(block.start_row, start - 1), "Operating_State"] = "startup_transient" if not summary["startup_not_captured"] else "unclassified"
            steady_mask = annotated.index.to_series().between(start, end)
            annotated.loc[steady_mask, "Operating_State"] = "steady_state"
            annotated.loc[steady_mask, "Is_Steady_State"] = True
            annotated.loc[annotated.index.to_series().between(end + 1, block.end_row), "Operating_State"] = "stopping_transient"
    return annotated
