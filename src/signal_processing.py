"""Configurable encoder processing for timing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import detrend, savgol_filter

from .test_segmentation import seconds_to_samples


@dataclass
class EncoderProcessingResult:
    median_filtered: np.ndarray
    smoothed: np.ndarray
    detrended: np.ndarray
    spike_flags: np.ndarray
    warnings: list[str]


def validated_odd_window(seconds: float, sampling_interval_s: float, sample_count: int, minimum: int = 3) -> int:
    """Convert seconds to a valid odd window no larger than the signal."""
    window = seconds_to_samples(seconds, sampling_interval_s, minimum)
    if window % 2 == 0:
        window += 1
    maximum = sample_count if sample_count % 2 else sample_count - 1
    return max(1, min(window, maximum))


def process_encoder(values: pd.Series, sampling_interval_s: float, config: dict[str, Any], estimated_cycle_s: float | None = None) -> EncoderProcessingResult:
    """Preserve raw values while producing spike-resistant timing signals."""
    settings = config.get("filtering", {})
    numeric = pd.to_numeric(values, errors="coerce")
    warnings: list[str] = []
    if numeric.notna().sum() < 3:
        raw = numeric.to_numpy(dtype=float)
        return EncoderProcessingResult(raw.copy(), raw.copy(), raw.copy(), np.zeros(len(raw), dtype=bool), ["Encoder run is too short or missing for filtering."])
    filled = numeric.interpolate(limit_direction="both")
    raw = filled.to_numpy(dtype=float)
    if not bool(settings.get("enabled", True)):
        warnings.append("Encoder filtering disabled; timing confidence is reduced.")
        return EncoderProcessingResult(raw.copy(), raw.copy(), raw.copy(), np.zeros(len(raw), dtype=bool), warnings)

    median_window = validated_odd_window(float(settings.get("median_window_s", 0.06)), sampling_interval_s, len(raw))
    median_filtered = filled.rolling(median_window, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual = raw - median_filtered
    residual_mad = float(np.median(np.abs(residual - np.median(residual))))
    threshold = float(settings.get("spike_mad_threshold", 6.0))
    scale = 1.4826 * residual_mad
    spike_flags = np.abs(residual) > threshold * scale if scale > 0 else np.abs(residual) > 0

    if estimated_cycle_s is not None and np.isfinite(estimated_cycle_s) and estimated_cycle_s > 0:
        savgol_seconds = estimated_cycle_s * float(settings.get("savgol_cycle_fraction", 0.12))
        savgol_seconds = min(float(settings.get("savgol_max_window_s", 1.0)), max(float(settings.get("savgol_min_window_s", 0.15)), savgol_seconds))
    else:
        savgol_seconds = float(settings.get("savgol_window_s", 0.60))
    savgol_window = validated_odd_window(savgol_seconds, sampling_interval_s, len(raw), 5)
    order = int(settings.get("savgol_polynomial_order", 3))
    if savgol_window <= order:
        warnings.append("Savitzky-Golay window is incompatible with polynomial order; median-filtered data used.")
        smoothed = median_filtered.copy()
    else:
        smoothed = savgol_filter(median_filtered, savgol_window, order, mode="interp")
    detrended_values = detrend(smoothed, type="linear") if bool(settings.get("detrend_enabled", False)) and len(smoothed) >= 3 else smoothed.copy()
    return EncoderProcessingResult(median_filtered, smoothed, detrended_values, spike_flags, warnings)
