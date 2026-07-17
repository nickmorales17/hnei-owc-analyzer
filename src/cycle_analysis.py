"""Final encoder behavior and multi-method cycle timing analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

from .signal_processing import EncoderProcessingResult
from .test_segmentation import seconds_to_samples


def classify_encoder_behavior(raw: pd.Series, processed: np.ndarray, sampling_interval_s: float) -> dict[str, Any]:
    """Classify likely encoder representation without inventing physical units."""
    values = pd.to_numeric(raw, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 10 or np.ptp(values) == 0:
        return {"encoder_behavior": "unusable", "classification_confidence": 0.0, "classification_basis": "Too few varying samples."}
    differences = np.diff(values)
    monotonic_fraction = max(np.mean(differences >= 0), np.mean(differences <= 0))
    unique_ratio = len(np.unique(values)) / len(values)
    large_jumps = np.abs(differences) > max(np.std(differences) * 5, np.ptp(values) * 0.25)
    quantized = np.mean(np.isclose(values, np.round(values))) > 0.95
    centered = processed - np.mean(processed)
    autocorr = correlate(centered, centered, mode="full", method="fft")[len(centered)-1:]
    periodic_strength = 0.0
    if autocorr[0] > 0:
        autocorr /= autocorr[0]
        minimum = seconds_to_samples(3.0, sampling_interval_s)
        peaks, _ = find_peaks(autocorr[minimum:])
        periodic_strength = float(np.max(autocorr[peaks + minimum])) if len(peaks) else 0.0
    if monotonic_fraction > 0.98 and np.sum(large_jumps) == 0:
        behavior = "unwrapped angular position" if quantized else "linear position"
    elif unique_ratio < 0.05 and np.mean(values == 0) > 0.5:
        behavior = "index or pulse signal"
    elif periodic_strength > 0.35 or (periodic_strength > 0.15 and np.sum(large_jumps) > 0):
        behavior = "processed periodic position" if np.sum(large_jumps) > 0 or not quantized else "wrapped angular position"
    else:
        behavior = "unknown periodic signal" if periodic_strength > 0.15 else "unusable"
    confidence = float(np.clip(0.45 + 0.5 * periodic_strength, 0, 0.95)) if behavior != "unusable" else 0.2
    basis = f"monotonic_fraction={monotonic_fraction:.3f}; periodic_strength={periodic_strength:.3f}; quantized={quantized}; narrow_jump_count={int(np.sum(large_jumps))}. Cycle timing uses waveform repetition and does not require counts per revolution."
    return {"encoder_behavior": behavior, "classification_confidence": confidence, "classification_basis": basis}


def _event_method(indices: np.ndarray, dt: float, name: str) -> dict[str, Any]:
    intervals = np.diff(indices) * dt
    if len(intervals) == 0:
        return {"method": name, "period_s": np.nan, "frequency_hz": np.nan, "event_interval_count": 0, "period_std_s": np.nan, "confidence": 0.0, "failure_reason": "Fewer than two events detected."}
    period = float(np.median(intervals))
    cv = float(np.std(intervals, ddof=0) / np.mean(intervals)) if np.mean(intervals) else np.inf
    return {"method": name, "period_s": period, "frequency_hz": 1 / period, "event_interval_count": len(intervals), "period_std_s": float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0, "confidence": float(np.clip(1 - cv * 3, 0, 1)), "failure_reason": ""}


def autocorrelation_period(values: np.ndarray, dt: float, minimum_s: float, maximum_s: float) -> dict[str, Any]:
    centered = values - np.mean(values)
    corr = correlate(centered, centered, mode="full", method="fft")[len(values)-1:]
    if len(corr) == 0 or corr[0] <= 0:
        return {"method": "autocorrelation", "period_s": np.nan, "frequency_hz": np.nan, "event_interval_count": 0, "period_std_s": np.nan, "confidence": 0.0, "failure_reason": "No autocorrelation energy."}
    corr /= corr[0]
    lo, hi = seconds_to_samples(minimum_s, dt), min(len(corr)-1, seconds_to_samples(maximum_s, dt))
    peaks, _ = find_peaks(corr[lo:hi+1])
    if not len(peaks):
        return {"method": "autocorrelation", "period_s": np.nan, "frequency_hz": np.nan, "event_interval_count": 0, "period_std_s": np.nan, "confidence": 0.0, "failure_reason": "No autocorrelation peak in configured range."}
    candidates = peaks + lo; lag = int(candidates[np.argmax(corr[candidates])]); period = lag * dt
    return {"method": "autocorrelation", "period_s": period, "frequency_hz": 1/period, "event_interval_count": 1, "period_std_s": np.nan, "confidence": float(np.clip(corr[lag], 0, 1)), "failure_reason": ""}


def fft_period(values: np.ndarray, dt: float, minimum_s: float, maximum_s: float) -> dict[str, Any]:
    centered = values - np.mean(values); frequencies = np.fft.rfftfreq(len(values), dt); power = np.abs(np.fft.rfft(centered)) ** 2
    mask = (frequencies >= 1/maximum_s) & (frequencies <= 1/minimum_s)
    if not np.any(mask) or np.sum(power[mask]) == 0:
        return {"method": "fft", "period_s": np.nan, "frequency_hz": np.nan, "event_interval_count": 0, "period_std_s": np.nan, "confidence": 0.0, "failure_reason": "No spectral energy in configured range."}
    indexes = np.flatnonzero(mask); index = indexes[np.argmax(power[indexes])]; frequency = frequencies[index]
    return {"method": "fft", "period_s": 1/frequency, "frequency_hz": frequency, "event_interval_count": 1, "period_std_s": np.nan, "confidence": float(power[index] / np.sum(power[mask])), "failure_reason": ""}


def zero_crossing_period(values: np.ndarray, dt: float, direction: str = "rising") -> dict[str, Any]:
    centered = values - np.median(values)
    if direction == "falling":
        crossings = np.flatnonzero((centered[:-1] >= 0) & (centered[1:] < 0)) + 1
    else:
        crossings = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0)) + 1
    return _event_method(crossings, dt, "zero_crossing")


def flag_interval_outliers(intervals: np.ndarray, tolerance: float) -> np.ndarray:
    if len(intervals) == 0:
        return np.zeros(0, dtype=bool)
    median = np.median(intervals)
    return np.abs(intervals - median) / median > tolerance


def analyze_encoder_cycles(run_id: str, steady: pd.DataFrame, processing: EncoderProcessingResult, sampling_interval_s: float, target_cycle_s: float, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    settings = config.get("encoder", {})
    expected = float(target_cycle_s)
    distance = seconds_to_samples(expected * float(settings.get("minimum_peak_spacing_fraction", 0.6)), sampling_interval_s)
    peaks, _ = find_peaks(processing.smoothed, prominence=float(settings.get("peak_prominence", 60)), distance=distance)
    troughs, _ = find_peaks(-processing.smoothed, prominence=float(settings.get("trough_prominence", 60)), distance=distance)
    methods = [
        _event_method(peaks, sampling_interval_s, "peak_to_peak"),
        _event_method(troughs, sampling_interval_s, "trough_to_trough"),
        autocorrelation_period(processing.detrended, sampling_interval_s, float(settings.get("autocorrelation_min_period_s", 1.5)), float(settings.get("autocorrelation_max_period_s", 10))),
        fft_period(processing.detrended, sampling_interval_s, float(settings.get("fft_min_period_s", 1.5)), float(settings.get("fft_max_period_s", 10))),
        zero_crossing_period(processing.detrended, sampling_interval_s, str(settings.get("zero_crossing_direction", "rising"))),
    ]
    method_table = pd.DataFrame(methods); method_table.insert(0, "Run_ID", run_id)
    event_periods = method_table.loc[method_table.method.isin(["peak_to_peak", "trough_to_trough", "zero_crossing"]) & method_table.period_s.notna() & (method_table.confidence >= 0.4), "period_s"]
    tolerance = float(settings.get("method_agreement_tolerance_fraction", 0.05))
    agreement = float((event_periods.max()-event_periods.min())/event_periods.median()) if len(event_periods) > 1 else np.nan
    reliable = method_table[(method_table.period_s.notna()) & (method_table.confidence >= 0.4)]
    if len(reliable) >= 2 and (not np.isfinite(agreement) or agreement <= tolerance):
        preferred = reliable[reliable.method.isin(["peak_to_peak", "trough_to_trough", "zero_crossing"])]
        selected_rows = preferred if not preferred.empty else reliable
        selected_period = float(np.average(selected_rows.period_s, weights=np.maximum(selected_rows.confidence, 0.01)))
        selected_method = "+".join(selected_rows.method.tolist())
        timing_confidence = float(np.clip(selected_rows.confidence.mean() * min(1, len(peaks)/4), 0, 1))
        selection_reason = "Confidence-weighted agreement of reliable cycle-level timing methods within the configured method-agreement tolerance."
    else:
        selected_period, selected_method, timing_confidence = np.nan, "unresolved", 0.0
        selection_reason = "No final period selected because reliable timing methods did not satisfy the configured agreement rule."

    # Use the event family with more complete intervals for the cycle-level table.
    cycle_events = troughs if len(troughs) > len(peaks) else peaks
    cycle_method = "trough_to_trough" if len(troughs) > len(peaks) else "peak_to_peak"
    intervals = np.diff(cycle_events) * sampling_interval_s
    outliers = flag_interval_outliers(intervals, float(settings.get("cycle_interval_deviation_tolerance", 0.10)))
    rows = []
    median_interval = float(np.median(intervals)) if len(intervals) else np.nan
    timestamps = steady["TimeStamp"].reset_index(drop=True)
    raw = steady["Encoder"].reset_index(drop=True)
    for number, (left, right, interval, outlier) in enumerate(zip(cycle_events[:-1], cycle_events[1:], intervals, outliers), 1):
        portion = raw.iloc[left:right+1]
        rows.append({"Run_ID": run_id, "Cycle_Number": number, "Cycle_Start_Time": timestamps.iloc[left], "Cycle_End_Time": timestamps.iloc[right], "Measured_Cycle_s": interval, "Measured_Frequency_Hz": 1/interval, "Peak_Encoder": float(portion.max()), "Minimum_Encoder": float(portion.min()), "Encoder_Amplitude": float(portion.max()-portion.min()), "Interval_Deviation_From_Run_Median_s": interval-median_interval, "Interval_Outlier_Flag": bool(outlier), "Timing_Method": cycle_method, "Start_Local_Index": int(left), "End_Local_Index": int(right)})
    cycle_table = pd.DataFrame(rows)
    valid = intervals[~outliers]
    minimum_valid = int(settings.get("minimum_valid_intervals", 4))
    warnings = []
    if len(valid) < minimum_valid:
        warnings.append(f"Fewer than {minimum_valid} valid cycle intervals; timing confidence is limited.")
    if timing_confidence < float(settings.get("timing_confidence_threshold", 0.60)):
        warnings.append("Timing confidence is below the configured reporting threshold.")
    warning = " ".join(warnings)
    def statistics(values: np.ndarray) -> dict[str, float]:
        if len(values) == 0:
            return {name: np.nan for name in ["mean", "median", "minimum", "maximum", "range", "population_std", "sample_std", "rms", "population_cv_ratio", "population_cv_percent", "sample_cv_ratio", "sample_cv_percent", "mean_frequency", "population_frequency_std", "sample_frequency_std"]}
        mean = float(np.mean(values))
        population_std = float(np.std(values, ddof=0))
        sample_std = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        population_cv = population_std / mean if mean else np.nan
        sample_cv = sample_std / mean if mean and np.isfinite(sample_std) else np.nan
        frequencies = 1 / values
        return {
            "mean": mean, "median": float(np.median(values)), "minimum": float(np.min(values)), "maximum": float(np.max(values)), "range": float(np.ptp(values)),
            "population_std": population_std, "sample_std": sample_std, "rms": float(np.sqrt(np.mean(values**2))),
            "population_cv_ratio": population_cv, "population_cv_percent": population_cv*100 if np.isfinite(population_cv) else np.nan,
            "sample_cv_ratio": sample_cv, "sample_cv_percent": sample_cv*100 if np.isfinite(sample_cv) else np.nan,
            "mean_frequency": float(np.mean(frequencies)), "population_frequency_std": float(np.std(frequencies, ddof=0)),
            "sample_frequency_std": float(np.std(frequencies, ddof=1)) if len(frequencies) >= 2 else np.nan,
        }
    all_stats, valid_stats = statistics(intervals), statistics(valid)
    summary = {
        "Run_ID": run_id,
        "All_Interval_Count": len(intervals), "Valid_Interval_Count": len(valid),
        "All_Interval_Mean_s": all_stats["mean"], "Valid_Interval_Mean_s": valid_stats["mean"],
        "All_Interval_Median_s": all_stats["median"], "Valid_Interval_Median_s": valid_stats["median"],
        "All_Minimum_Cycle_s": all_stats["minimum"], "Valid_Minimum_Cycle_s": valid_stats["minimum"],
        "All_Maximum_Cycle_s": all_stats["maximum"], "Valid_Maximum_Cycle_s": valid_stats["maximum"],
        "All_Peak_To_Peak_Range_s": all_stats["range"], "Valid_Peak_To_Peak_Range_s": valid_stats["range"],
        "All_Population_Std_Cycle_s": all_stats["population_std"], "All_Sample_Std_Cycle_s": all_stats["sample_std"],
        "Valid_Population_Std_Cycle_s": valid_stats["population_std"], "Valid_Sample_Std_Cycle_s": valid_stats["sample_std"],
        "All_RMS_Cycle_s": all_stats["rms"], "Valid_RMS_Cycle_s": valid_stats["rms"],
        "All_Population_CV_Ratio": all_stats["population_cv_ratio"], "All_Population_CV_Percent": all_stats["population_cv_percent"],
        "All_Sample_CV_Ratio": all_stats["sample_cv_ratio"], "All_Sample_CV_Percent": all_stats["sample_cv_percent"],
        "Valid_Population_CV_Ratio": valid_stats["population_cv_ratio"], "Valid_Population_CV_Percent": valid_stats["population_cv_percent"],
        "Valid_Sample_CV_Ratio": valid_stats["sample_cv_ratio"], "Valid_Sample_CV_Percent": valid_stats["sample_cv_percent"],
        "All_Mean_Measured_Frequency_Hz": all_stats["mean_frequency"], "Valid_Mean_Measured_Frequency_Hz": valid_stats["mean_frequency"],
        "All_Population_Frequency_Std_Hz": all_stats["population_frequency_std"], "All_Sample_Frequency_Std_Hz": all_stats["sample_frequency_std"],
        "Valid_Population_Frequency_Std_Hz": valid_stats["population_frequency_std"], "Valid_Sample_Frequency_Std_Hz": valid_stats["sample_frequency_std"],
        "Final_Selected_Period_s": selected_period, "Final_Selected_Frequency_Hz": 1/selected_period if np.isfinite(selected_period) else np.nan,
        "Selected_Timing_Method": selected_method, "Selection_Reason": selection_reason, "Method_Agreement_Fraction": agreement,
        "Timing_Method_Confidence": timing_confidence, "Low_Cycle_Count_Warning": warning,
    }
    return cycle_table, method_table, summary, {"peaks": peaks, "troughs": troughs}
