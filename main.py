"""Command-line entry point for the HNEI OWC Stage 1–3 pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

from src.data_validation import validate_data
from src.file_loader import load_config, load_data_file
from src.plotting import create_stage3_diagnostics
from src.steady_state import annotate_operating_states, classify_steady_state
from src.test_segmentation import (
    blocks_to_table,
    detect_active_blocks,
    estimate_preliminary_period,
    group_recorded_targets,
    match_expected_period,
)
from src.utilities import prepare_output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit HNEI OWC CSV or Excel test data.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX/XLS path")
    parser.add_argument("--output", help="New or empty output directory")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--sheet", help="Excel worksheet name")
    parser.add_argument("--file-type", default="auto", choices=["auto", "csv", "xlsx", "xls"])
    parser.add_argument("--show-plots", action="store_true", help="Plots are saved; interactive display remains disabled")
    parser.add_argument("--no-smoothing", action="store_true", help="Reserved for later signal-analysis stages")
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
    logging.info("Starting Stage 1–3 intake and detection pipeline")
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
    sampling_interval = float(validated.audit["sampling_interval_median_s"])
    blocks = group_recorded_targets(validated.data, float(config.get("target_cycle_tolerance_s", 0.1)))
    if not blocks:
        blocks = detect_active_blocks(validated.data, sampling_interval, config)
    logging.info("Detected %d operating blocks", len(blocks))

    period_rows: list[dict[str, object]] = []
    detection_results = {}
    for block in blocks:
        if block.target_source == "recorded":
            estimate = estimate_preliminary_period(validated.data.loc[block.start_row:block.end_row, "Encoder"], sampling_interval, config) if "Encoder" in validated.data else {}
        else:
            estimate = estimate_preliminary_period(validated.data.loc[block.start_row:block.end_row, "Encoder"], sampling_interval, config)
            label, confidence = match_expected_period(float(estimate["selected_period_s"]), float(estimate["period_confidence"]), config)
            block.provisional_target_cycle_s = label
            block.detection_confidence = confidence
            if label is None:
                block.detection_warnings = "Estimated target label did not meet confidence threshold."
        period_rows.append({
            "run_id": block.run_id,
            "peak_period_s": estimate.get("peak_period_s"),
            "autocorrelation_period_s": estimate.get("autocorrelation_period_s"),
            "method_difference_s": estimate.get("method_difference_s"),
            "selected_preliminary_period_s": estimate.get("selected_period_s"),
            "period_confidence": estimate.get("period_confidence"),
            "provisional_target_cycle_s": block.provisional_target_cycle_s,
            "detection_confidence": block.detection_confidence,
            "target_source": block.target_source,
        })
        detection_results[block.run_id] = classify_steady_state(validated.data, block, sampling_interval, config)
        if detection_results[block.run_id][1]["startup_not_captured"]:
            message = "File/run begins during active operation; startup was not captured."
            block.detection_warnings = "; ".join(filter(None, [block.detection_warnings, message]))

    annotated = annotate_operating_states(validated.data, blocks, detection_results)
    annotated.to_csv(output_dir / "cleaned" / "full_annotated_data.csv", index=False)
    annotated.loc[annotated["Is_Steady_State"]].to_csv(output_dir / "cleaned" / "steady_state_data.csv", index=False)
    for block in blocks:
        run_data = annotated.loc[block.start_row:block.end_row]
        run_data.to_csv(output_dir / "cleaned" / f"{block.run_id}_all_data.csv", index=False)
        run_data.loc[run_data["Is_Steady_State"]].to_csv(output_dir / "cleaned" / f"{block.run_id}_steady_state.csv", index=False)

    write_audit(validated.audit, output_dir / "tables" / "intake_audit.csv")
    pd.DataFrame(validated.findings).to_csv(output_dir / "tables" / "data_quality_findings.csv", index=False)
    boundaries = blocks_to_table(blocks, annotated)
    period_table = pd.DataFrame(period_rows)
    steady_table = pd.DataFrame([detection_results[block.run_id][1] for block in blocks])
    cycle_tables = [detection_results[block.run_id][0] for block in blocks if not detection_results[block.run_id][0].empty]
    boundaries.to_csv(output_dir / "tables" / "run_boundaries.csv", index=False)
    period_table.to_csv(output_dir / "tables" / "preliminary_period_estimates.csv", index=False)
    steady_table.to_csv(output_dir / "tables" / "steady_state_selection.csv", index=False)
    pd.concat(cycle_tables, ignore_index=True).to_csv(output_dir / "tables" / "cycle_classification.csv", index=False) if cycle_tables else pd.DataFrame().to_csv(output_dir / "tables" / "cycle_classification.csv", index=False)
    graphs = create_stage3_diagnostics(annotated, blocks, detection_results, output_dir, config)
    logging.info("Created %d Stage 3 diagnostic graphs", len(graphs))

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
    print(f"Median-derived sampling rate: {validated.audit['median_derived_sampling_rate_hz']:.6g} Hz")
    print(f"Effective mean sampling rate: {validated.audit['effective_mean_sampling_rate_hz']:.6g} Hz")
    print(f"Sampling-interval CV: {validated.audit['sampling_interval_cv_ratio']:.6g} ({validated.audit['sampling_interval_cv_percent']:.3f}%)")
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
    print("Detected runs:")
    for block, period in zip(blocks, period_rows):
        summary = detection_results[block.run_id][1]
        print(f"- {block.run_id}: rows {block.start_row}-{block.end_row}, preliminary period={period['selected_preliminary_period_s']:.3f} s, inferred target={block.provisional_target_cycle_s}, confidence={block.detection_confidence:.3f}, steady cycles={summary['steady_cycle_count']}")
    print(f"Output directory: {output_dir}")
    logging.info("Stage 1–3 pipeline complete")
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
