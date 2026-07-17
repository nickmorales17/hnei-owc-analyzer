"""Command-line entry point for the HNEI OWC Stage 1–5 pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.data_validation import validate_data
from src.file_loader import load_config, load_data_file
from src.cycle_analysis import analyze_encoder_cycles, classify_encoder_behavior
from src.plotting import create_stage3_diagnostics, create_stage4_diagnostics, create_stage5_diagnostics
from src.pressure_analysis import add_derived_pressure_channels, add_dynamic_pressure_channels, analyze_pressure_pairs, pressure_response_summary
from src.statistics_analysis import descriptive_statistics, cycle_level_statistics, torque_summary, generator_summary, correlation_regression_summary
from src.correlation_analysis import torque_phase_summary
from src.quality_checks import apply_quality_checks
from src.signal_processing import process_encoder
from src.steady_state import annotate_operating_states, classify_steady_state
from src.test_segmentation import (
    blocks_to_table,
    detect_active_blocks,
    estimate_preliminary_period,
    group_recorded_targets,
    match_expected_period,
)
from src.utilities import prepare_output_directory
from src.vfd_analysis import verify_vfd_run


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
    logging.info("Starting Stage 1–5 intake, timing, VFD, and response-analysis pipeline")
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
    preliminary_by_run: dict[str, float] = {}
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
        preliminary_by_run[block.run_id] = float(estimate.get("selected_period_s", float("nan")))
        detection_results[block.run_id] = classify_steady_state(validated.data, block, sampling_interval, config)
        if detection_results[block.run_id][1]["startup_not_captured"]:
            message = "File/run begins during active operation; startup was not captured."
            block.detection_warnings = "; ".join(filter(None, [block.detection_warnings, message]))

    annotated = annotate_operating_states(validated.data, blocks, detection_results)
    annotated["Encoder_Median_Filtered"] = np.nan
    annotated["Encoder_Smoothed"] = np.nan
    annotated["Encoder_Detrended"] = np.nan
    annotated["Encoder_Spike_Flag"] = False
    annotated["Final_Cycle_Number"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["Cycle_Interval_Outlier_Flag"] = False
    for column in ["Reconstructed_Target_VFD_Hz","Reconstructed_Command_mV_Uncapped","Reconstructed_Command_mV_Capped","Command_Equivalent_Frequency_Hz","Command_Equivalent_Cycle_s"]:
        annotated[column] = np.nan
    annotated["Command_Saturation_Flag"] = False

    behavior_rows=[]; final_cycle_tables={}; method_tables=[]; summary_rows=[]; vfd_rows=[]; final_events={}
    for block in blocks:
        run_index=annotated.loc[block.start_row:block.end_row].index
        target=float(block.provisional_target_cycle_s) if block.provisional_target_cycle_s is not None else float("nan")
        timing_reference = target if np.isfinite(target) and target > 0 else preliminary_by_run[block.run_id]
        run_processing=process_encoder(annotated.loc[run_index,"Encoder"],sampling_interval,config,timing_reference)
        annotated.loc[run_index,"Encoder_Median_Filtered"]=run_processing.median_filtered
        annotated.loc[run_index,"Encoder_Smoothed"]=run_processing.smoothed
        annotated.loc[run_index,"Encoder_Detrended"]=run_processing.detrended
        annotated.loc[run_index,"Encoder_Spike_Flag"]=run_processing.spike_flags
        behavior={"Run_ID":block.run_id,**classify_encoder_behavior(annotated.loc[run_index,"Encoder"],run_processing.smoothed,sampling_interval),"spike_count":int(run_processing.spike_flags.sum()),"processing_warnings":"; ".join(run_processing.warnings)}; behavior_rows.append(behavior)
        steady=annotated.loc[run_index][annotated.loc[run_index,"Is_Steady_State"]]
        steady_processing=process_encoder(steady["Encoder"],sampling_interval,config,timing_reference)
        cycles,methods,summary,events=analyze_encoder_cycles(block.run_id,steady,steady_processing,sampling_interval,timing_reference,config)
        final_cycle_tables[block.run_id]=cycles; method_tables.append(methods); summary["Nominal_Target_Cycle_s"]=target; summary["Target_Source"]=block.target_source; summary_rows.append(summary); final_events[block.run_id]=events
        if not cycles.empty:
            steady_rows=steady.index.to_numpy()
            for _,cycle in cycles.iterrows():
                rows=steady_rows[int(cycle.Start_Local_Index):int(cycle.End_Local_Index)+1]
                annotated.loc[rows,"Final_Cycle_Number"]=int(cycle.Cycle_Number); annotated.loc[rows,"Cycle_Interval_Outlier_Flag"]=bool(cycle.Interval_Outlier_Flag)
        if np.isfinite(target) and target > 0:
            vfd=verify_vfd_run(block.run_id,annotated.loc[run_index],target,block.target_source,float(summary["Final_Selected_Period_s"]),config,sampling_interval,int(summary["Valid_Interval_Count"]),float(summary["Valid_Peak_To_Peak_Range_s"]))
        else:
            final_period=summary["Final_Selected_Period_s"]
            vfd={"Run_ID":block.run_id,"Nominal_Target_Cycle_s":np.nan,"Target_Source":"unclassified","VFD_Verification_Status":"unavailable_no_target","Desired_Target_Frequency_Hz":np.nan,"Reconstructed_Command_mV_Uncapped":np.nan,"Reconstructed_Command_mV_Capped":np.nan,"Command_Equivalent_Frequency_Hz":np.nan,"Command_Equivalent_Cycle_s":np.nan,"Command_Saturation_Flag":False,"Final_Measured_Cycle_s":final_period,"Final_Measured_Cycle_Frequency_Hz":1/final_period if np.isfinite(final_period) else np.nan,"Final_Measured_VFD_Equivalent_Frequency_Hz":np.nan,"Command_Interpretation":"VFD verification unavailable because no recorded or confidently inferred target is available."}
        vfd_rows.append(vfd)
        annotated.loc[run_index,"Reconstructed_Target_VFD_Hz"]=vfd["Desired_Target_Frequency_Hz"]
        annotated.loc[run_index,"Reconstructed_Command_mV_Uncapped"]=vfd["Reconstructed_Command_mV_Uncapped"]
        annotated.loc[run_index,"Reconstructed_Command_mV_Capped"]=vfd["Reconstructed_Command_mV_Capped"]
        annotated.loc[run_index,"Command_Equivalent_Frequency_Hz"]=vfd["Command_Equivalent_Frequency_Hz"]
        annotated.loc[run_index,"Command_Equivalent_Cycle_s"]=vfd["Command_Equivalent_Cycle_s"]
        annotated.loc[run_index,"Command_Saturation_Flag"]=vfd["Command_Saturation_Flag"]

    final_method_table=pd.concat(method_tables,ignore_index=True); final_summary_table=pd.DataFrame(summary_rows); vfd_table=pd.DataFrame(vfd_rows); behavior_table=pd.DataFrame(behavior_rows); final_intervals=pd.concat([table for table in final_cycle_tables.values() if not table.empty],ignore_index=True)
    annotated,derived_channels,derived_skipped=add_derived_pressure_channels(annotated)
    annotated,quality_findings,quality_counts=apply_quality_checks(annotated,sampling_interval,config)
    annotated,dynamic_channels=add_dynamic_pressure_channels(annotated); derived_channels.extend(dynamic_channels)
    run_periods={str(row.Run_ID):float(row.Final_Selected_Period_s) for _,row in final_summary_table.iterrows()}
    descriptive_table=descriptive_statistics(annotated)
    cycle_level_table=cycle_level_statistics(annotated,config)
    pressure_pair_table,pressure_consistency_table=analyze_pressure_pairs(annotated,run_periods,sampling_interval,config)
    pressure_response_table=pressure_response_summary(annotated,cycle_level_table,config)
    torque_table=torque_summary(annotated,cycle_level_table,run_periods)
    torque_phase_table=torque_phase_summary(annotated,run_periods,sampling_interval,config)
    generator_table=generator_summary(annotated,run_periods,config)
    correlation_table=correlation_regression_summary(annotated,cycle_level_table,torque_table,generator_table,pressure_response_table,int(config.get("stage5",{}).get("minimum_regression_observations",3)))
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
    behavior_table.to_csv(output_dir/"tables"/"encoder_behavior_classification.csv",index=False)
    final_intervals.to_csv(output_dir/"tables"/"encoder_cycle_intervals.csv",index=False)
    final_method_table.to_csv(output_dir/"tables"/"encoder_cycle_method_comparison.csv",index=False)
    final_summary_table.to_csv(output_dir/"tables"/"encoder_cycle_summary.csv",index=False)
    vfd_table.to_csv(output_dir/"tables"/"vfd_command_verification.csv",index=False)
    descriptive_table.to_csv(output_dir/"tables"/"descriptive_statistics.csv",index=False)
    cycle_level_table.to_csv(output_dir/"tables"/"cycle_level_statistics.csv",index=False)
    pressure_response_table.to_csv(output_dir/"tables"/"pressure_response_summary.csv",index=False)
    pressure_pair_table.to_csv(output_dir/"tables"/"pressure_pair_relationships.csv",index=False)
    pressure_pair_table[[c for c in ["Run_ID","Signal_1","Signal_2","Data_State","Data_Version","Zero_Lag_Pearson","Zero_Lag_Spearman","Maximum_Lagged_Correlation","Maximum_Absolute_Lagged_Correlation","Signed_Lag_Samples","Signed_Lag_Seconds","Phase_Degrees","Wrapped_Phase_Degrees","Lag_Sign_Convention","Measured_Cycle_s","Lag_Search_Limit_s","Reliability","Reliability_Reason"] if c in pressure_pair_table]].to_csv(output_dir/"tables"/"pressure_phase_lag_summary.csv",index=False)
    pressure_consistency_table.to_csv(output_dir/"tables"/"pressure_sensor_consistency.csv",index=False)
    torque_table.to_csv(output_dir/"tables"/"torque_summary.csv",index=False)
    torque_phase_table.to_csv(output_dir/"tables"/"torque_phase_summary.csv",index=False)
    generator_table.to_csv(output_dir/"tables"/"generator_voltage_summary.csv",index=False)
    correlation_table.to_csv(output_dir/"tables"/"correlation_regression_summary.csv",index=False)
    quality_findings.to_csv(output_dir/"tables"/"quality_flags_summary.csv",index=False)
    graphs = create_stage3_diagnostics(annotated, blocks, detection_results, output_dir, config)
    logging.info("Created %d Stage 3 diagnostic graphs", len(graphs))
    stage4_graphs=create_stage4_diagnostics(annotated,blocks,final_cycle_tables,final_method_table,final_summary_table,vfd_table,final_events,output_dir,config)
    logging.info("Created %d Stage 4 diagnostic graphs",len(stage4_graphs))
    stage5_graphs=create_stage5_diagnostics(annotated,blocks,pressure_pair_table,pressure_response_table,torque_table,generator_table,quality_counts,output_dir,config)
    logging.info("Created %d Stage 5 diagnostic graphs",len(stage5_graphs))

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
    print("Final encoder cycle and VFD verification:")
    for summary,vfd in zip(summary_rows,vfd_rows):
        capped = f"{vfd['Command_Equivalent_Cycle_s']:.3f} s" if pd.notna(vfd['Command_Equivalent_Cycle_s']) else "unavailable"
        print(f"- {summary['Run_ID']}: final selected={summary['Final_Selected_Period_s']:.3f} s, valid interval mean={summary['Valid_Interval_Mean_s']:.3f} s, valid sample std={summary['Valid_Sample_Std_Cycle_s']:.4f} s, valid sample CV={summary['Valid_Sample_CV_Ratio']:.6f} ({summary['Valid_Sample_CV_Percent']:.3f}%), timing-method confidence={summary['Timing_Method_Confidence']:.3f}, capped expectation={capped}, saturation={vfd['Command_Saturation_Flag']}")
    print("Stage 5 response analysis:")
    print(f"- Derived pressure channels: {', '.join(derived_channels) if derived_channels else 'none'}")
    for skipped in derived_skipped: print(f"- Skipped: {skipped}")
    print(f"- Pressure relationships evaluated: {len(pressure_pair_table)}; quality findings: {len(quality_findings)}")
    print("- Primary torque trend: torque increased with measured cycle frequency and decreased with commanded cycle period; the n=4 run-summary regression is exploratory.")
    for _,row in pressure_response_table.iterrows(): print(f"- {row.Run_ID} attenuation: raw={row.Raw_Attenuation_Ratio:.3f}, robust={row.Robust_Attenuation_Ratio:.3f}, median-cycle={row.Median_Cycle_Attenuation_Ratio:.3f}, raw/robust disagreement={row.Raw_Robust_Attenuation_Disagreement_Flag}")
    for _,row in generator_table.iterrows(): print(f"- {row.Run_ID} Gen_V: drift={row.Drift_Classification}, periodicity={row.Periodicity_Classification}, combined={row.Combined_GenV_Classification}")
    print(f"Output directory: {output_dir}")
    logging.info("Stage 1–5 pipeline complete")
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
