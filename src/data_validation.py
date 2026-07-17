"""Timestamp parsing and non-destructive intake validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Annotated data, audit metrics, and immediate findings."""

    data: pd.DataFrame
    audit: dict[str, Any]
    findings: list[dict[str, str]]
    warnings: list[str]


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse native/text, Unix, or Excel-serial timestamps without unsafe guessing."""
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
    """Calculate consecutive intervals in seconds."""
    return timestamps.diff().dt.total_seconds()


def _add_flag(flags: pd.Series, mask: pd.Series, label: str) -> None:
    indexes = flags.index[mask.fillna(False)]
    flags.loc[indexes] = flags.loc[indexes].map(lambda value: f"{value};{label}" if value else label)


def validate_data(frame: pd.DataFrame, config: dict[str, Any]) -> ValidationResult:
    """Annotate questionable rows and calculate an intake audit."""
    data = frame.copy()
    required = list(config.get("required_columns", ["TimeStamp"]))
    missing_required = [column for column in required if column not in data.columns]
    if missing_required:
        raise ValueError(f"Required columns unavailable after alias mapping: {missing_required}")

    warnings: list[str] = []
    findings: list[dict[str, str]] = []
    flags = pd.Series("", index=data.index, dtype="object")
    data["TimeStamp"] = parse_timestamps(data["TimeStamp"])
    timestamp_parse_failures = int(data["TimeStamp"].isna().sum())
    _add_flag(flags, data["TimeStamp"].isna(), "timestamp_parse_failure")

    numeric_failures: dict[str, int] = {}
    for column in config.get("numeric_columns", []):
        if column not in data.columns:
            continue
        original_nonmissing = data[column].notna()
        converted = pd.to_numeric(data[column], errors="coerce")
        failure_mask = original_nonmissing & converted.isna()
        numeric_failures[column] = int(failure_mask.sum())
        _add_flag(flags, failure_mask, f"numeric_conversion_failure:{column}")
        data[column] = converted

    timestamp_reversals = data["TimeStamp"].diff().dt.total_seconds().lt(0)
    reversal_count = int(timestamp_reversals.sum())
    _add_flag(flags, timestamp_reversals, "timestamp_reversal")
    if reversal_count:
        data["Quality_Flags"] = flags
        data = data.sort_values("TimeStamp", kind="stable", na_position="last").reset_index(drop=True)
        flags = data["Quality_Flags"].copy()
        warnings.append("Timestamp reversals were detected; working data were stably sorted by timestamp.")

    duplicate_row_mask = data.drop(columns=["Original_Row_Order"], errors="ignore").duplicated(keep=False)
    duplicate_timestamp_mask = data["TimeStamp"].notna() & data["TimeStamp"].duplicated(keep=False)
    duplicate_record_mask = pd.Series(False, index=data.index)
    if "RecNum" in data:
        duplicate_record_mask = data["RecNum"].notna() & data["RecNum"].duplicated(keep=False)
    _add_flag(flags, duplicate_row_mask, "duplicate_row")
    _add_flag(flags, duplicate_timestamp_mask, "duplicate_timestamp")
    _add_flag(flags, duplicate_record_mask, "duplicate_record_number")

    record_gap_count = 0
    if "RecNum" in data:
        record_diff = data["RecNum"].diff()
        record_gap_mask = record_diff.gt(1)
        record_gap_count = int(record_gap_mask.sum())
        _add_flag(flags, record_gap_mask, "record_number_gap_after_previous_row")

    intervals = calculate_sampling_intervals(data["TimeStamp"])
    positive_intervals = intervals[intervals > 0]
    median_interval = float(positive_intervals.median()) if not positive_intervals.empty else np.nan
    gap_factor = float(config.get("validation", {}).get("timing_gap_factor", 3.0))
    timing_gap_mask = intervals.gt(median_interval * gap_factor) if np.isfinite(median_interval) else pd.Series(False, index=data.index)
    _add_flag(flags, timing_gap_mask, "timing_gap_after_previous_row")
    data["Sampling_Interval_s"] = intervals
    data["Quality_Flags"] = flags

    missing_by_column = data.drop(
        columns=["Quality_Flags", "Sampling_Interval_s"], errors="ignore"
    ).isna().sum()
    total_missing = int(missing_by_column.sum())
    jitter_threshold = float(config.get("validation", {}).get("sampling_jitter_relative_threshold", 0.10))
    jitter_cv = (
        float(positive_intervals.std(ddof=0) / positive_intervals.mean())
        if len(positive_intervals) > 1 and positive_intervals.mean() != 0
        else np.nan
    )
    if total_missing:
        details = ", ".join(f"{key}={value}" for key, value in missing_by_column.items() if value)
        findings.append({"severity": "warning", "check": "missing_values", "finding": details})
    if reversal_count:
        findings.append({"severity": "warning", "check": "timestamp_reversals", "finding": str(reversal_count)})
    if int(timing_gap_mask.sum()):
        findings.append({"severity": "warning", "check": "timing_gaps", "finding": str(int(timing_gap_mask.sum()))})
    if np.isfinite(jitter_cv) and jitter_cv > jitter_threshold:
        findings.append({"severity": "warning", "check": "sampling_jitter", "finding": f"CV={jitter_cv:.6g}"})
    for name, count in numeric_failures.items():
        if count:
            findings.append({"severity": "warning", "check": f"numeric_conversion:{name}", "finding": str(count)})

    valid_timestamps = data["TimeStamp"].dropna()
    audit: dict[str, Any] = {
        "row_count": len(data),
        "column_count": len([column for column in frame.columns if column != "Original_Row_Order"]),
        "timestamp_start": valid_timestamps.min() if not valid_timestamps.empty else None,
        "timestamp_end": valid_timestamps.max() if not valid_timestamps.empty else None,
        "duration_s": (valid_timestamps.max() - valid_timestamps.min()).total_seconds() if len(valid_timestamps) > 1 else np.nan,
        "timestamp_parse_failures": timestamp_parse_failures,
        "missing_value_count": total_missing,
        "duplicate_row_count": int(duplicate_row_mask.sum()),
        "duplicate_timestamp_count": int(duplicate_timestamp_mask.sum()),
        "duplicate_record_number_count": int(duplicate_record_mask.sum()),
        "record_gap_count": record_gap_count,
        "timestamp_reversal_count": reversal_count,
        "timing_gap_count": int(timing_gap_mask.sum()),
        "sampling_interval_median_s": median_interval,
        "sampling_interval_mean_s": float(positive_intervals.mean()) if not positive_intervals.empty else np.nan,
        "sampling_interval_min_s": float(positive_intervals.min()) if not positive_intervals.empty else np.nan,
        "sampling_interval_max_s": float(positive_intervals.max()) if not positive_intervals.empty else np.nan,
        "sampling_interval_cv": jitter_cv,
        "estimated_sampling_rate_hz": 1.0 / median_interval if np.isfinite(median_interval) and median_interval > 0 else np.nan,
    }
    for column, count in missing_by_column.items():
        audit[f"missing:{column}"] = int(count)
    for column, count in numeric_failures.items():
        audit[f"numeric_conversion_failures:{column}"] = count
    if not findings:
        findings.append({"severity": "info", "check": "intake", "finding": "No immediate configured data-quality issues detected."})
    return ValidationResult(data, audit, findings, warnings)
