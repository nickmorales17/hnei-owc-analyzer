"""Command-line entry point for the HNEI OWC Stage 1–2 intake audit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

from src.data_validation import validate_data
from src.file_loader import load_config, load_data_file
from src.utilities import prepare_output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit HNEI OWC CSV or Excel test data.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX/XLS path")
    parser.add_argument("--output", help="New or empty output directory")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--sheet", help="Excel worksheet name")
    parser.add_argument("--file-type", default="auto", choices=["auto", "csv", "xlsx", "xls"])
    parser.add_argument("--show-plots", action="store_true", help="Reserved for Stage 3+")
    parser.add_argument("--no-smoothing", action="store_true", help="Reserved for Stage 3+")
    parser.add_argument("--debug", action="store_true")
    return parser


def configure_logging(log_path: Path, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def write_audit(audit: dict[str, object], path: Path) -> None:
    pd.DataFrame([{"metric": key, "value": value} for key, value in audit.items()]).to_csv(path, index=False)


def run(args: argparse.Namespace) -> Path:
    output_dir = prepare_output_directory(args.output)
    configure_logging(output_dir / "run_log.txt", args.debug)
    logging.info("Starting Stage 1–2 intake audit")
    config = load_config(args.config)
    loaded = load_data_file(args.input, config, sheet=args.sheet, file_type=args.file_type)
    logging.info("Detected source columns: %s", ", ".join(loaded.source_columns))
    if loaded.available_sheets:
        logging.info("Available worksheets: %s", ", ".join(loaded.available_sheets))
        logging.info("Selected worksheet: %s", loaded.selected_sheet)
    for message in loaded.assumptions:
        logging.info("Assumption: %s", message)
    for message in loaded.warnings:
        logging.warning(message)

    validated = validate_data(loaded.data, config)
    for message in validated.warnings:
        logging.warning(message)
    validated.data.to_csv(output_dir / "cleaned" / "full_annotated_data.csv", index=False)
    write_audit(validated.audit, output_dir / "tables" / "intake_audit.csv")
    pd.DataFrame(validated.findings).to_csv(output_dir / "tables" / "data_quality_findings.csv", index=False)

    print("\nINTAKE AUDIT")
    print(f"Detected columns: {', '.join(column for column in loaded.data.columns if column != 'Original_Row_Order')}")
    print(f"Timestamp range: {validated.audit['timestamp_start']} to {validated.audit['timestamp_end']}")
    print(
        "Sampling interval (s): "
        f"median={validated.audit['sampling_interval_median_s']:.9g}, "
        f"mean={validated.audit['sampling_interval_mean_s']:.9g}, "
        f"min={validated.audit['sampling_interval_min_s']:.9g}, "
        f"max={validated.audit['sampling_interval_max_s']:.9g}"
    )
    print(f"Estimated sampling rate: {validated.audit['estimated_sampling_rate_hz']:.6g} Hz")
    print(f"Missing values: {validated.audit['missing_value_count']}")
    print(
        "Duplicates: "
        f"rows={validated.audit['duplicate_row_count']}, "
        f"timestamps={validated.audit['duplicate_timestamp_count']}, "
        f"record_numbers={validated.audit['duplicate_record_number_count']}"
    )
    print(f"Record gaps: {validated.audit['record_gap_count']}")
    print(f"Timestamp reversals: {validated.audit['timestamp_reversal_count']}")
    print("Immediate findings:")
    for finding in validated.findings:
        print(f"- [{finding['severity']}] {finding['check']}: {finding['finding']}")
    print(f"Output directory: {output_dir}")
    logging.info("Stage 1–2 intake audit complete")
    return output_dir


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        logging.error("Intake audit failed: %s", exc)
        if args.debug:
            logging.exception("Debug traceback")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

