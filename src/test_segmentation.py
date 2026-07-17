"""Run grouping, activity detection, and preliminary period estimation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import correlate, find_peaks, savgol_filter


@dataclass
class RunBlock:
    run_id: str
    start_row: int
    end_row: int
    target_source: str
    provisional_target_cycle_s: float | None = None
    detection_confidence: float = 0.0
    signals_used: str = ""
    detection_warnings: str = ""


def seconds_to_samples(seconds: float, sampling_interval_s: float, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds / sampling_interval_s)))


def group_recorded_targets(data: pd.DataFrame, tolerance_s: float) -> list[RunBlock]:
    """Split every contiguous recorded target sequence, including repeated targets."""
    if "Target_Cycle_s" not in data:
        return []
    targets = pd.to_numeric(data["Target_Cycle_s"], errors="coerce")
    blocks: list[RunBlock] = []
    start: int | None = None
    reference = np.nan
    for position, value in enumerate(targets):
        active = pd.notna(value) and float(value) > 0
        same = active and start is not None and abs(float(value) - float(reference)) <= tolerance_s
        if active and start is None:
            start, reference = position, float(value)
        elif active and not same:
            blocks.append(RunBlock(f"run_{len(blocks)+1:03d}", start, position - 1, "recorded", float(reference), 1.0, "Target_Cycle_s"))
            start, reference = position, float(value)
        elif not active and start is not None:
            blocks.append(RunBlock(f"run_{len(blocks)+1:03d}", start, position - 1, "recorded", float(reference), 1.0, "Target_Cycle_s"))
            start, reference = None, np.nan
    if start is not None:
        blocks.append(RunBlock(f"run_{len(blocks)+1:03d}", start, len(data) - 1, "recorded", float(reference), 1.0, "Target_Cycle_s"))
    return blocks


def _mask_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def detect_active_blocks(data: pd.DataFrame, sampling_interval_s: float, config: dict[str, Any]) -> list[RunBlock]:
    """Detect active blocks from rolling encoder and supporting-signal activity."""
    settings = config.get("activity_detection", {})
    manual = config.get("manual_run_boundary_overrides", [])
    if manual:
        return [
            RunBlock(
                str(item.get("run_id", f"run_{index:03d}")), int(item["start_row"]), int(item["end_row"]),
                str(item.get("target_source", "manual")), item.get("target_cycle_s"), 1.0,
                "manual_override", "Manual run-boundary override applied.",
            )
            for index, item in enumerate(manual, 1)
        ]

    window = seconds_to_samples(float(settings.get("rolling_window_s", 1.0)), sampling_interval_s, 3)
    if window % 2 == 0:
        window += 1
    scores = np.zeros(len(data), dtype=float)
    used: list[str] = []
    if "Encoder" in data and data["Encoder"].notna().sum() >= window:
        encoder = data["Encoder"].astype(float)
        rolling_std = encoder.rolling(window, center=True, min_periods=window // 2).std()
        rolling_range = encoder.rolling(window, center=True, min_periods=window // 2).max() - encoder.rolling(window, center=True, min_periods=window // 2).min()
        encoder_active = (rolling_std >= float(settings.get("encoder_std_threshold", 10.0))) | (rolling_range >= float(settings.get("encoder_range_threshold", 35.0)))
        scores += encoder_active.fillna(False).to_numpy(dtype=float)
        used.append("Encoder")

    noise_multiplier = float(settings.get("supporting_noise_multiplier", 5.0))
    for signal in settings.get("signals", ["Pressure_1", "Pressure_2", "Torque"]):
        if signal == "Encoder" or signal not in data or data[signal].notna().sum() < window:
            continue
        rolling = data[signal].astype(float).rolling(window, center=True, min_periods=window // 2).std()
        baseline = float(rolling.quantile(0.1))
        if np.isfinite(baseline) and baseline > 0:
            scores += 0.35 * (rolling > baseline * noise_multiplier).fillna(False).to_numpy(dtype=float)
            used.append(signal)
    if not used:
        raise ValueError("Active-run detection requires at least one usable configured signal.")

    mask = scores >= float(settings.get("active_score_threshold", 1.0))
    merge_seconds = max(
        float(settings.get("block_merge_gap_s", 0.75)),
        float(settings.get("minimum_idle_duration_s", 1.5)),
    )
    # Binary closing expands from both sides, so half the desired gap width is used.
    merge = seconds_to_samples(merge_seconds / 2.0, sampling_interval_s)
    cleanup = seconds_to_samples(float(settings.get("edge_cleanup_s", 0.25)), sampling_interval_s)
    mask = binary_closing(mask, iterations=merge)
    mask = binary_opening(mask, iterations=cleanup)
    minimum = seconds_to_samples(float(settings.get("minimum_active_duration_s", 8.0)), sampling_interval_s)
    raw_blocks = [(start, end) for start, end in _mask_blocks(mask) if end - start + 1 >= minimum]
    # Centered rolling windows can trim an already-active file edge. Extend only
    # when the first detected block is within one window and the leading encoder
    # independently satisfies the configured activity rules.
    if raw_blocks and raw_blocks[0][0] <= window and "Encoder" in data:
        leading = data["Encoder"].iloc[:window].astype(float)
        leading_active = (
            leading.std() >= float(settings.get("encoder_std_threshold", 10.0))
            or leading.max() - leading.min() >= float(settings.get("encoder_range_threshold", 35.0))
        )
        if leading_active:
            raw_blocks[0] = (0, raw_blocks[0][1])
    return [RunBlock(f"run_{index:03d}", start, end, "inferred", signals_used=", ".join(used)) for index, (start, end) in enumerate(raw_blocks, 1)]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window if window % 2 else window + 1, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window, min(3, window - 2))


def estimate_preliminary_period(values: pd.Series, sampling_interval_s: float, config: dict[str, Any]) -> dict[str, Any]:
    """Estimate repetition period using peak timing and autocorrelation."""
    settings = config.get("period_estimation", {})
    raw = pd.to_numeric(values, errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    expected = [float(value) for value in config.get("expected_cycle_times_s", []) if float(value) > 0]
    reference_period = min(expected) if expected else float(settings.get("autocorrelation_min_period_s", 1.5))
    smoothing_seconds = float(settings.get("smoothing_window_s", reference_period * float(settings.get("smoothing_cycle_fraction", 0.12))))
    smoothing_seconds = min(float(settings.get("maximum_smoothing_window_s", 1.0)), max(float(settings.get("minimum_smoothing_window_s", 0.15)), smoothing_seconds))
    window = seconds_to_samples(smoothing_seconds, sampling_interval_s, 5)
    processed = _smooth(raw, window)
    spacing_seconds = float(settings.get("minimum_peak_spacing_s", reference_period * float(settings.get("minimum_peak_spacing_fraction", 0.55))))
    distance = seconds_to_samples(spacing_seconds, sampling_interval_s)
    prominence = settings.get("peak_prominence", 60.0)
    peaks, _ = find_peaks(processed, prominence=prominence, distance=distance)
    peak_period = float(np.median(np.diff(peaks)) * sampling_interval_s) if len(peaks) >= 2 else np.nan

    centered = processed - np.mean(processed)
    correlation = correlate(centered, centered, mode="full", method="fft")[len(centered) - 1 :]
    if correlation[0] > 0:
        correlation = correlation / correlation[0]
    min_lag = seconds_to_samples(float(settings.get("autocorrelation_min_period_s", 1.5)), sampling_interval_s)
    max_lag = min(len(correlation) - 1, seconds_to_samples(float(settings.get("autocorrelation_max_period_s", 10.0)), sampling_interval_s))
    local_peaks, _ = find_peaks(correlation[min_lag : max_lag + 1])
    if len(local_peaks):
        candidates = local_peaks + min_lag
        best_lag = int(candidates[np.argmax(correlation[candidates])])
        autocorr_period = best_lag * sampling_interval_s
        autocorr_strength = float(max(0.0, correlation[best_lag]))
    else:
        autocorr_period, autocorr_strength = np.nan, 0.0
    available = [value for value in (peak_period, autocorr_period) if np.isfinite(value)]
    if available and expected:
        # Expected targets guide inference only: choose the independently measured
        # method nearest a configured target instead of averaging a harmonic with
        # a fundamental period.
        selected = min(available, key=lambda value: min(abs(value-target) for target in expected))
        selected_target_distance = min(abs(selected-target) for target in expected)
    else:
        selected = float(np.median(available)) if available else np.nan
        selected_target_distance = np.nan
    difference = abs(peak_period - autocorr_period) if np.isfinite(peak_period) and np.isfinite(autocorr_period) else np.nan
    method_agreement = max(0.0, 1.0 - difference / 1.0) if np.isfinite(difference) else 0.35 if available else 0.0
    confidence = float(np.clip(0.6 * method_agreement + 0.4 * autocorr_strength, 0, 1))
    if np.isfinite(selected_target_distance):
        target_tolerance = float(settings.get("expected_match_tolerance_s", 0.75))
        confidence = max(confidence, float(np.clip(0.5 + 0.5*(1-selected_target_distance/target_tolerance), 0, 1)))
    return {
        "peak_period_s": peak_period,
        "autocorrelation_period_s": autocorr_period,
        "method_difference_s": difference,
        "selected_period_s": selected,
        "period_confidence": confidence,
        "peak_indices": peaks,
        "processed_signal": processed,
    }


def match_expected_period(estimated_s: float, period_confidence: float, config: dict[str, Any]) -> tuple[float | None, float]:
    if not np.isfinite(estimated_s):
        return None, 0.0
    expected = np.asarray(config.get("expected_cycle_times_s", [8, 7, 6, 5]), dtype=float)
    closest = float(expected[np.argmin(abs(expected - estimated_s))])
    tolerance = float(config.get("period_estimation", {}).get("expected_match_tolerance_s", 0.75))
    match_confidence = max(0.0, 1.0 - abs(closest - estimated_s) / tolerance)
    confidence = float(np.clip(0.5 * period_confidence + 0.5 * match_confidence, 0, 1))
    threshold = float(config.get("inferred_label_confidence_threshold", 0.65))
    return (closest if confidence >= threshold else None), confidence


def blocks_to_table(blocks: list[RunBlock], data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in blocks:
        row = asdict(block)
        row.update({
            "start_timestamp": data.loc[block.start_row, "TimeStamp"],
            "end_timestamp": data.loc[block.end_row, "TimeStamp"],
            "duration_s": (data.loc[block.end_row, "TimeStamp"] - data.loc[block.start_row, "TimeStamp"]).total_seconds(),
            "sample_count": block.end_row - block.start_row + 1,
        })
        rows.append(row)
    return pd.DataFrame(rows)
