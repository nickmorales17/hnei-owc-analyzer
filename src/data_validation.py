"""Time-base selection and non-destructive intake validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    data: pd.DataFrame
    audit: dict[str, Any]
    findings: list[dict[str, str]]
    warnings: list[str]


def parse_timestamps(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_fraction >= 0.9 and numeric.notna().any():
        median = float(numeric.dropna().median())
        if 20_000 <= median <= 80_000:
            return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        if 1e9 <= median <= 5e9:
            return pd.to_datetime(numeric, unit="s", origin="unix", errors="coerce")
        if 1e12 <= median <= 5e12:
            return pd.to_datetime(numeric, unit="ms", origin="unix", errors="coerce")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def calculate_sampling_intervals(timestamps: pd.Series) -> pd.Series:
    return timestamps.diff().dt.total_seconds()


def _add_flag(flags: pd.Series, mask: pd.Series, label: str) -> None:
    indexes = flags.index[mask.fillna(False)]
    flags.loc[indexes] = flags.loc[indexes].map(lambda value: f"{value};{label}" if value else label)


def validate_data(frame: pd.DataFrame, config: dict[str, Any], sample_interval_s: float | None = None) -> ValidationResult:
    """Choose TimeStamp, numeric elapsed time, or validated record reconstruction."""
    data = frame.copy()
    required = [c for c in config.get("required_columns", []) if c != "TimeStamp"]
    missing_required = [c for c in required if c not in data]
    if missing_required:
        raise ValueError(f"Required columns unavailable after alias mapping: {missing_required}")

    warnings: list[str] = []
    findings: list[dict[str, str]] = []
    flags = pd.Series("", index=data.index, dtype="object")
    numeric_failures: dict[str, int] = {}
    for column in config.get("numeric_columns", []):
        if column not in data:
            continue
        original_nonmissing = data[column].notna()
        converted = pd.to_numeric(data[column], errors="coerce")
        failure_mask = original_nonmissing & converted.isna()
        numeric_failures[column] = int(failure_mask.sum())
        _add_flag(flags, failure_mask, f"numeric_conversion_failure:{column}")
        data[column] = converted

    timestamp_parse_failures = 0
    reversal_count = 0
    duplicate_timestamp_mask = pd.Series(False, index=data.index)
    assumed_interval: float | None = None
    if "TimeStamp" in data:
        data["TimeStamp"] = parse_timestamps(data["TimeStamp"])
        timestamp_parse_failures = int(data["TimeStamp"].isna().sum())
        if timestamp_parse_failures:
            raise ValueError(f"TimeStamp contains {timestamp_parse_failures} unavailable or unparseable value(s); cannot establish a complete time axis.")
        time_source = "recorded_timestamp"
        elapsed = calculate_sampling_intervals(data["TimeStamp"]).fillna(0).cumsum()
        timestamp_reversals = data["TimeStamp"].diff().dt.total_seconds().lt(0)
        reversal_count = int(timestamp_reversals.sum())
        _add_flag(flags, timestamp_reversals, "timestamp_reversal")
        if reversal_count:
            data["Quality_Flags"] = flags
            data = data.sort_values("TimeStamp", kind="stable").reset_index(drop=True)
            flags = data["Quality_Flags"].copy()
            elapsed = (data["TimeStamp"] - data["TimeStamp"].iloc[0]).dt.total_seconds()
            warnings.append("Timestamp reversals were detected; working data were stably sorted by timestamp.")
        duplicate_timestamp_mask = data["TimeStamp"].duplicated(keep=False)
    elif "Elapsed_Time_s" in data:
        elapsed = pd.to_numeric(data["Elapsed_Time_s"], errors="coerce")
        if elapsed.isna().any():
            raise ValueError("Elapsed_Time_s must be numeric with no missing values.")
        if elapsed.diff().dropna().le(0).any():
            raise ValueError("Elapsed_Time_s must be monotonically increasing with no duplicate values.")
        time_source = "recorded_elapsed_time"
    else:
        assumed_interval = sample_interval_s
        if assumed_interval is None:
            assumed_interval = config.get("intake", {}).get("fallback_sample_interval_s")
        if assumed_interval is None:
            raise ValueError("TimeStamp is unavailable. Supply --sample-interval-s or configure intake.fallback_sample_interval_s.")
        assumed_interval = float(assumed_interval)
        if not np.isfinite(assumed_interval) or assumed_interval <= 0:
            raise ValueError("The fallback sample interval must be a finite value greater than zero.")
        if "RecNum" not in data:
            raise ValueError("TimeStamp is unavailable and RecNum is required for time reconstruction.")
        rec = pd.to_numeric(data["RecNum"], errors="coerce")
        if rec.isna().any():
            raise ValueError("RecNum must be numeric with no missing values for time reconstruction.")
        if rec.duplicated().any():
            raise ValueError("RecNum contains duplicate record numbers; time reconstruction cannot continue.")
        if rec.diff().dropna().le(0).any():
            raise ValueError("RecNum resets or reverses; record numbers must be monotonically increasing for time reconstruction.")
        elapsed = (rec - rec.iloc[0]) * assumed_interval
        time_source = "reconstructed_from_record_number"
        warnings.append(f"Elapsed time was reconstructed from RecNum using an assumed fixed interval of {assumed_interval:g} s; timing-dependent results depend on this assumption.")

    data["Elapsed_Time_s"] = elapsed.astype(float)
    data["Time_Source"] = time_source
    data["Time_Reconstructed_Flag"] = time_source == "reconstructed_from_record_number"
    data["Assumed_Sample_Interval_s"] = assumed_interval if assumed_interval is not None else np.nan

    duplicate_row_mask = data.drop(columns=["Original_Row_Order"], errors="ignore").duplicated(keep=False)
    duplicate_record_mask = pd.Series(False, index=data.index)
    record_gap_mask = pd.Series(False, index=data.index)
    if "RecNum" in data:
        duplicate_record_mask = data["RecNum"].notna() & data["RecNum"].duplicated(keep=False)
        record_gap_mask = data["RecNum"].diff().gt(1)
    _add_flag(flags, duplicate_row_mask, "duplicate_row")
    _add_flag(flags, duplicate_timestamp_mask, "duplicate_timestamp")
    _add_flag(flags, duplicate_record_mask, "duplicate_record_number")
    _add_flag(flags, record_gap_mask, "record_number_gap_after_previous_row")

    intervals = data["Elapsed_Time_s"].diff()
    positive_intervals = intervals[intervals > 0]
    median_interval = float(positive_intervals.median()) if not positive_intervals.empty else np.nan
    gap_factor = float(config.get("validation", {}).get("timing_gap_factor", 3.0))
    timing_gap_mask = intervals.gt(median_interval * gap_factor) if np.isfinite(median_interval) else pd.Series(False, index=data.index)
    _add_flag(flags, timing_gap_mask, "timing_gap_after_previous_row")
    data["Sampling_Interval_s"] = intervals
    data["Quality_Flags"] = flags

    missing_by_column = data.drop(columns=["Quality_Flags", "Sampling_Interval_s", "Assumed_Sample_Interval_s"], errors="ignore").isna().sum()
    total_missing = int(missing_by_column.sum())
    jitter_available = time_source == "recorded_timestamp"
    jitter_cv = float(positive_intervals.std(ddof=0) / positive_intervals.mean()) if jitter_available and len(positive_intervals) > 1 and positive_intervals.mean() else np.nan
    if total_missing:
        findings.append({"severity":"warning","check":"missing_values","finding":", ".join(f"{k}={v}" for k,v in missing_by_column.items() if v)})
    if reversal_count:
        findings.append({"severity":"warning","check":"timestamp_reversals","finding":str(reversal_count)})
    if int(timing_gap_mask.sum()):
        findings.append({"severity":"warning","check":"timing_gaps","finding":str(int(timing_gap_mask.sum()))})
    if not jitter_available:
        findings.append({"severity":"info","check":"sampling_jitter","finding":"unavailable; timing jitter cannot be evaluated without recorded timestamps"})
    for name, count in numeric_failures.items():
        if count: findings.append({"severity":"warning","check":f"numeric_conversion:{name}","finding":str(count)})

    valid_timestamps = data["TimeStamp"] if "TimeStamp" in data else pd.Series(dtype="datetime64[ns]")
    duration = float(data["Elapsed_Time_s"].iloc[-1] - data["Elapsed_Time_s"].iloc[0]) if len(data) > 1 else 0.0
    audit: dict[str, Any] = {
        "row_count":len(data), "column_count":len([c for c in frame if c != "Original_Row_Order"]),
        "time_source":time_source, "time_reconstructed_flag":time_source == "reconstructed_from_record_number",
        "assumed_sample_interval_s":assumed_interval, "timestamp_start":valid_timestamps.min() if len(valid_timestamps) else None,
        "timestamp_end":valid_timestamps.max() if len(valid_timestamps) else None, "absolute_timestamp_range":"unavailable" if not len(valid_timestamps) else f"{valid_timestamps.min()} to {valid_timestamps.max()}",
        "duration_s":duration, "timestamp_parse_failures":timestamp_parse_failures, "missing_value_count":total_missing,
        "duplicate_row_count":int(duplicate_row_mask.sum()), "duplicate_timestamp_count":int(duplicate_timestamp_mask.sum()),
        "duplicate_record_number_count":int(duplicate_record_mask.sum()), "record_gap_count":int(record_gap_mask.sum()),
        "timestamp_reversal_count":reversal_count, "timing_gap_count":int(timing_gap_mask.sum()),
        "sampling_interval_median_s":median_interval, "sampling_interval_mean_s":float(positive_intervals.mean()) if len(positive_intervals) else np.nan,
        "sampling_interval_min_s":float(positive_intervals.min()) if len(positive_intervals) else np.nan, "sampling_interval_max_s":float(positive_intervals.max()) if len(positive_intervals) else np.nan,
        "sampling_jitter_available":jitter_available, "sampling_interval_cv_ratio":jitter_cv, "sampling_interval_cv_percent":jitter_cv*100 if np.isfinite(jitter_cv) else np.nan,
        "median_derived_sampling_rate_hz":1/median_interval if median_interval > 0 else np.nan,
        "effective_mean_sampling_rate_hz":1/positive_intervals.mean() if len(positive_intervals) and positive_intervals.mean() > 0 else np.nan,
        "timing_dependency_note":"Lag, phase, frequency, period, and drift-per-second results depend on the assumed fixed interval." if time_source == "reconstructed_from_record_number" else "not applicable",
    }
    for column,count in missing_by_column.items(): audit[f"missing:{column}"] = int(count)
    for column,count in numeric_failures.items(): audit[f"numeric_conversion_failures:{column}"] = count
    if not findings: findings.append({"severity":"info","check":"intake","finding":"No immediate configured data-quality issues detected."})
    return ValidationResult(data,audit,findings,warnings)
