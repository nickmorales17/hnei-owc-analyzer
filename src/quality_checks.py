"""Non-destructive Stage 5 signal-quality checks and filtered companions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress


SIGNALS = ["Pressure_1", "Pressure_2", "Pressure_3", "Pressure_4", "Torque", "Gen_V"]


def robust_spike_flags(values: pd.Series, sampling_interval_s: float, config: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    settings = config.get("stage5", {})
    window = max(3, int(round(float(settings.get("quality_rolling_window_s", .25)) / sampling_interval_s)))
    if window % 2 == 0:
        window += 1
    numeric = pd.to_numeric(values, errors="coerce")
    baseline = numeric.rolling(window, center=True, min_periods=max(2, window // 3)).median()
    residual = numeric - baseline
    median = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - median)))
    scale = 1.4826 * mad
    if scale > 0:
        severity = residual.abs() / scale
    else:
        tolerance = max(float(np.nanstd(numeric)) * 1e-9, 1e-12)
        severity = pd.Series(np.where(residual.abs() > tolerance, np.inf, 0.0), index=values.index)
    flags = severity > float(settings.get("spike_mad_threshold", 8.0))
    derivative_limit = settings.get("derivative_spike_threshold_per_s")
    if derivative_limit is not None:
        flags |= numeric.diff().abs().div(sampling_interval_s) > float(derivative_limit)
    filtered = numeric.mask(flags, baseline)
    return flags.fillna(False).astype(bool), severity.fillna(0.0), filtered


def _finding(run_id: object, signal: str, check: str, severity: str, evidence: str) -> dict[str, object]:
    return {"Run_ID": run_id, "Signal": signal, "Check": check, "Severity": severity, "Evidence": evidence}


def apply_quality_checks(data: pd.DataFrame, sampling_interval_s: float, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annotate quality evidence while leaving recorded source channels unchanged."""
    result = data.copy(); findings: list[dict[str, object]] = []
    settings = config.get("stage5", {})
    for signal in SIGNALS:
        if signal not in result:
            continue
        flags, severity, filtered = robust_spike_flags(result[signal], sampling_interval_s, config)
        result[f"{signal}_Spike_Flag"] = flags
        result[f"{signal}_Spike_Severity"] = severity
        result[f"{signal}_Filtered"] = filtered
    for run_id, run in result[result.Run_ID.notna() & result.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        for signal in SIGNALS:
            if signal not in run:
                continue
            x = pd.to_numeric(run[signal], errors="coerce").dropna()
            if x.empty:
                continue
            spike_count = int(result.loc[x.index, f"{signal}_Spike_Flag"].sum())
            if spike_count:
                findings.append(_finding(run_id, signal, "isolated_spikes", "warning", f"{spike_count} robust rolling-MAD spike candidates; raw values retained"))
            counts = x.round(int(settings.get("clipping_decimal_places", 9))).value_counts()
            repeated_fraction = float(counts.iloc[0] / len(x))
            if repeated_fraction >= float(settings.get("clipping_repetition_fraction", .05)) and len(x) >= 20:
                findings.append(_finding(run_id, signal, "possible_clipping_or_saturation", "review", f"most repeated value occupies {repeated_fraction:.1%} of samples"))
            std = float(x.std(ddof=0))
            if std <= float(settings.get("near_constant_std", 1e-6)):
                findings.append(_finding(run_id, signal, "near_constant_signal", "review", f"population standard deviation={std:.6g}"))
            if len(x) >= 3:
                t = (result.loc[x.index, "Elapsed_Time_s"] - result.loc[x.index[0], "Elapsed_Time_s"]) if "Elapsed_Time_s" in result else (result.loc[x.index, "TimeStamp"] - result.loc[x.index[0], "TimeStamp"]).dt.total_seconds()
                slope = float(linregress(t, x).slope) if np.ptp(t) > 0 else np.nan
                fitted_change = slope * float(t.iloc[-1] - t.iloc[0]) if np.isfinite(slope) else np.nan
                scale = max(float(np.ptp(x)), abs(float(x.mean())), 1e-12)
                if np.isfinite(fitted_change) and abs(fitted_change) / scale >= float(settings.get("drift_relative_threshold", .1)):
                    findings.append(_finding(run_id, signal, "possible_drift", "review", f"linear fitted change={fitted_change:.6g}; slope={slope:.6g}/s"))
        if {"Pressure_3", "Pressure_4"}.issubset(run.columns):
            offset = float((run.Pressure_4 - run.Pressure_3).mean())
            if abs(offset) >= float(settings.get("pair_offset_tolerance", 50.0)):
                findings.append(_finding(run_id, "Pressure_4", "pressure_pair_offset", "review", f"mean Pressure_4 - Pressure_3 offset={offset:.6g}; evidence is not proof of failure"))
    finding_table = pd.DataFrame(findings, columns=["Run_ID", "Signal", "Check", "Severity", "Evidence"])
    counts = finding_table.groupby(["Run_ID", "Signal", "Check", "Severity"]).size().reset_index(name="Finding_Count") if not finding_table.empty else pd.DataFrame(columns=["Run_ID", "Signal", "Check", "Severity", "Finding_Count"])
    return result, finding_table, counts
