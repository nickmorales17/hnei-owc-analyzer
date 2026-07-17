"""CSV and Excel loading for the Stage 1–2 intake pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .column_mapping import map_columns, normalize_column_name


@dataclass
class LoadResult:
    """Loaded data and decisions made during intake."""

    data: pd.DataFrame
    source_columns: list[str]
    available_sheets: list[str]
    selected_sheet: str | None
    units_row_detected: bool
    units: dict[str, str]
    assumptions: list[str]
    warnings: list[str]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / str(parent)).resolve()
        config = _deep_merge(load_config(parent_path), config)
    if not isinstance(config.get("column_aliases"), dict):
        raise ValueError("Configuration must define a 'column_aliases' mapping.")
    return config


def _looks_like_units_row(row: pd.Series, config: dict[str, Any]) -> bool:
    nonempty = row.dropna()
    if nonempty.empty:
        return False
    text_count = 0
    for value in nonempty:
        text = str(value).strip()
        try:
            float(text)
        except ValueError:
            text_count += 1
    threshold = float(config.get("validation", {}).get("units_row_min_text_fraction", 0.5))
    return text_count / len(nonempty) >= threshold


def _header_score(columns: list[object], aliases: dict[str, list[str]]) -> int:
    accepted = {
        normalize_column_name(value)
        for canonical, values in aliases.items()
        for value in [canonical, *values]
    }
    return sum(normalize_column_name(column) in accepted for column in columns)


def _choose_excel_sheet(
    path: Path, requested_sheet: str | None, aliases: dict[str, list[str]]
) -> tuple[list[str], str]:
    workbook = pd.ExcelFile(path)
    sheets = workbook.sheet_names
    if not sheets:
        raise ValueError(f"Excel workbook has no worksheets: {path}")
    if requested_sheet:
        if requested_sheet not in sheets:
            raise ValueError(
                f"Worksheet '{requested_sheet}' not found. Available worksheets: {sheets}"
            )
        return sheets, requested_sheet
    scored: list[tuple[int, int, str]] = []
    for sheet in sheets:
        preview = pd.read_excel(path, sheet_name=sheet, nrows=10)
        score = _header_score(list(preview.columns), aliases)
        scored.append((score, len(preview), sheet))
    return sheets, max(scored)[2]


def load_data_file(
    path: str | Path,
    config: dict[str, Any],
    *,
    sheet: str | None = None,
    file_type: str = "auto",
) -> LoadResult:
    """Load CSV/Excel data, detect units, and apply configured aliases."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    requested_type = file_type.casefold()
    inferred_type = input_path.suffix.casefold().lstrip(".") if requested_type == "auto" else requested_type
    if inferred_type not in {"csv", "xlsx", "xls"}:
        raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")

    available_sheets: list[str] = []
    selected_sheet: str | None = None
    aliases = config["column_aliases"]
    if inferred_type == "csv":
        frame = pd.read_csv(input_path)
    else:
        available_sheets, selected_sheet = _choose_excel_sheet(input_path, sheet, aliases)
        frame = pd.read_excel(input_path, sheet_name=selected_sheet)

    source_columns = [str(column).strip() for column in frame.columns]
    frame.columns = source_columns
    units_row_detected = len(frame) > 0 and _looks_like_units_row(frame.iloc[0], config)
    units: dict[str, str] = {}
    if units_row_detected:
        units = {
            column: str(value).strip()
            for column, value in frame.iloc[0].items()
            if pd.notna(value)
        }
        frame = frame.iloc[1:].reset_index(drop=True)

    mapping = map_columns(frame.columns, aliases)
    frame = frame.rename(columns=mapping.rename_map)
    frame.insert(0, "Original_Row_Order", range(len(frame)))
    warnings = list(mapping.warnings)
    assumptions = list(mapping.assumptions)
    if selected_sheet and sheet is None:
        assumptions.append(f"Selected worksheet '{selected_sheet}' using header recognition.")
    if units_row_detected:
        assumptions.append("Detected and excluded a likely units row from data rows.")
    return LoadResult(
        frame,
        source_columns,
        available_sheets,
        selected_sheet,
        units_row_detected,
        units,
        assumptions,
        warnings,
    )
