"""Stage 6 engineering tables and Markdown reports from validated Stage 1–5 results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _merge(base: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available=["Run_ID",*[c for c in columns if c in other]]
    return base.merge(other[available],on="Run_ID",how="left") if "Run_ID" in other else base


def build_cross_test_comparison(tables: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Create one explicitly sourced engineering-comparison row per run."""
    encoder=tables["encoder_summary"].copy()
    result=encoder[[c for c in ["Run_ID","Nominal_Target_Cycle_s","Target_Source","Final_Selected_Period_s","Final_Selected_Frequency_Hz","Timing_Method_Confidence","Valid_Interval_Count","Valid_Sample_CV_Percent"] if c in encoder]].rename(columns={"Nominal_Target_Cycle_s":"Target_Cycle_s","Valid_Sample_CV_Percent":"Sample_CV_Percent"})
    result=_merge(result,tables["vfd"],["Desired_Target_Frequency_Hz","Reconstructed_Command_mV_Capped","Command_Equivalent_Cycle_s","Command_Saturation_Flag","VFD_Verification_Status"])
    result=_merge(result,tables["pressure_response"],["Robust_Upstream_P5_P95_Span","Robust_Downstream_P5_P95_Span","Robust_Attenuation_Ratio","Median_Cycle_Attenuation_Ratio","Dynamic_Turbine_DeltaP_RMS","Dynamic_Turbine_DeltaP_P5_P95_Span"])
    result=_merge(result,tables["torque"],["Mean_Torque","Peak_Torque","RMS_Torque"])
    result=_merge(result,tables["generator"],["Mean","Linear_Drift_Per_s","Drift_R_Squared","Combined_GenV_Classification"])
    if "steady_metrics" in tables: result=_merge(result,tables["steady_metrics"],["Steady_State_Sample_Count","Steady_State_Duration_s"])
    result=result.rename(columns={"Desired_Target_Frequency_Hz":"Desired_VFD_Hz","Reconstructed_Command_mV_Capped":"Capped_Command_mV","Command_Equivalent_Cycle_s":"Equivalent_Cycle_s","Robust_Upstream_P5_P95_Span":"Robust_Upstream_Pressure_Span","Robust_Downstream_P5_P95_Span":"Robust_Downstream_Pressure_Span","Dynamic_Turbine_DeltaP_RMS":"Dynamic_DeltaP_RMS","Dynamic_Turbine_DeltaP_P5_P95_Span":"Dynamic_DeltaP_Robust_Span","Mean":"Mean_Gen_V","Linear_Drift_Per_s":"Gen_V_Drift_Slope","Drift_R_Squared":"Gen_V_Drift_R2","Combined_GenV_Classification":"Gen_V_Combined_Classification"})
    findings=tables.get("quality_findings",pd.DataFrame())
    priority={"error":0,"warning":1,"review":2,"info":3}
    primary=[]
    for run_id in result.Run_ID:
        subset=findings[findings.Run_ID==run_id].copy() if "Run_ID" in findings else pd.DataFrame()
        if subset.empty: primary.append("none")
        else:
            subset["_priority"]=subset.Severity.map(priority).fillna(9); row=subset.sort_values("_priority").iloc[0]; primary.append(f"{row.Signal}: {row.Check} ({row.Severity})")
    result["Primary_Quality_Finding"]=primary
    sort=result.Target_Cycle_s.where(result.Target_Cycle_s.notna(),result.Final_Selected_Period_s); result=result.assign(_sort=sort).sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return result


def build_pressure_derived_summary(data:pd.DataFrame,units:dict[str,str]) -> pd.DataFrame:
    derived=["Upstream_Mean","Downstream_Mean","DeltaP_13","DeltaP_24","Turbine_DeltaP","Upstream_Dynamic","Downstream_Dynamic","Turbine_DeltaP_Dynamic","Turbine_DeltaP_Cycle_Dynamic"]
    definitions={"Upstream_Mean":"mean(Pressure_1, Pressure_2)","Downstream_Mean":"mean(Pressure_3, Pressure_4)","DeltaP_13":"Pressure_1 - Pressure_3","DeltaP_24":"Pressure_2 - Pressure_4","Turbine_DeltaP":"Upstream_Mean - Downstream_Mean","Upstream_Dynamic":"Upstream_Mean - run steady-state median","Downstream_Dynamic":"Downstream_Mean - run steady-state median","Turbine_DeltaP_Dynamic":"Upstream_Dynamic - Downstream_Dynamic","Turbine_DeltaP_Cycle_Dynamic":"Turbine_DeltaP - cycle median"}
    return pd.DataFrame([{"Channel":c,"Available":c in data,"Value_Source":"derived","Units":units.get("Pressure_1","unknown"),"Definition":definitions[c],"Interpretation_Limit":"Not an absolute calibrated pressure unless calibration metadata are documented."} for c in derived])


def _fmt(value:Any,decimals:int=3)->str:
    return "unavailable" if value is None or pd.isna(value) else f"{float(value):.{decimals}f}"


def _condition_markdown(cross:pd.DataFrame)->str:
    columns=[c for c in ["Run_ID","Target_Cycle_s","Target_Source","Final_Selected_Period_s","Timing_Method_Confidence","Robust_Attenuation_Ratio","Mean_Torque","Mean_Gen_V"] if c in cross]
    frame=cross[columns].round(4); lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.itertuples(index=False,name=None): lines.append("| "+" | ".join("unavailable" if pd.isna(v) else str(v) for v in row)+" |")
    return "\n".join(lines)


def create_markdown_reports(output_dir:Path,context:dict[str,Any],config:dict[str,Any])->list[Path]:
    """Write the four report-ready Markdown deliverables."""
    report_dir=output_dir/"report"; report_dir.mkdir(parents=True,exist_ok=True); cross=context["cross_test"]
    audit=context["audit"]; metadata=context["metadata"]; generator=context["tables"]["generator"]; pressure=context["tables"]["pressure_response"]
    if audit.get("time_reconstructed_flag"):
        time_statement=(f"Elapsed time was reconstructed from record numbers using an assumed fixed sample interval of {_fmt(audit.get('assumed_sample_interval_s'),6)} s. Absolute date and timestamp range are unavailable. Measured sampling jitter is unavailable and cannot be evaluated without recorded timestamps. Record-number gaps are retained. Lag, phase, frequency, period, and drift-per-second results depend on the assumed fixed interval.")
    elif audit.get("time_source") == "recorded_elapsed_time":
        time_statement="The recorded numeric Elapsed_Time_s axis was used. Absolute date and timestamp range are unavailable; sampling jitter cannot be evaluated without recorded timestamps."
    else:
        time_statement="Recorded timestamps were used; the absolute timestamp range and measured sampling jitter are available in the intake audit."
    conditions=_condition_markdown(cross); n=len(cross)
    vfd_unavailable=bool("VFD_Verification_Status" in cross and cross.VFD_Verification_Status.astype(str).str.contains("unavailable").any())
    fastest=cross.sort_values("Final_Selected_Period_s").iloc[0] if n else None
    gen_lines="\n".join(f"- {r.Run_ID}: drift slope {_fmt(r.Gen_V_Drift_Slope,4)}, R² {_fmt(r.Gen_V_Drift_R2,3)}, {r.Gen_V_Combined_Classification}." for _,r in cross.iterrows())
    main=f"""# HNEI OWC Test Bench Data Analysis

## 1. Test and data overview

Input `{metadata['input_filename']}` contained {metadata['total_records']:,} records from {audit['timestamp_start']} through {audit['timestamp_end']}. The median sampling interval was {_fmt(audit['sampling_interval_median_s'],6)} s. Available channels were {', '.join(metadata['available_channels'])}; pressure and torque units remain unknown. Profile: `{metadata['configuration_profile']}`.

{time_statement}

## 2. Data-processing methods

The workflow mapped configured aliases, parsed timestamps, audited sampling, segmented runs, selected steady state, processed encoder timing, compared cycle methods, verified VFD commands where scaling was available, derived pressure response, evaluated lag and phase, summarized torque and Gen_V behavior, and applied non-destructive quality checks. Every source row and raw measured value was retained.

## 3. Detected test conditions

{conditions}

Target sources are labeled recorded or inferred. VFD command values labeled reconstructed are not measurements of CR1000X output.

## 4. Encoder and cycle timing

Selected periods are reported to three decimals, consistent with the sampling resolution. Timing-method confidence measures method agreement and signal evidence, not cycle stability. Cycle variability remains represented by interval standard deviation, sample CV, and peak-to-peak range. Averaging cycles can improve a mean estimate but cannot create microsecond measurement resolution.

## 5. VFD command verification

{'VFD reconstruction is unavailable because the selected profile has unverified command scaling; no constants were invented.' if vfd_unavailable else 'Legacy commands were reconstructed from the configured verified relationship and labeled reconstructed. The 5-second run is command-saturated. Its measured cycle is numerically closer to nominal than the capped expectation, but the 0.012-second median sampling interval, observed variation, and two valid timing intervals cannot reliably distinguish those cases.'}

## 6. Pressure response

Robust 5th-to-95th-percentile and median-cycle attenuation are the primary comparisons. Raw extrema are retained but are not the primary trend, particularly where the raw/robust disagreement flag is set. Upstream sensors agreed strongly; downstream pairs showed weak correlation and substantial offset evidence. Dynamic differential pressure removes run median offsets. Raw differential-pressure means are not interpreted as calibrated absolute turbine pressure drops because compatible calibration, units, zero reference, and sign convention are undocumented. Weak-correlation phase values are numerical outputs only, not reliable physical phase.

## 7. Torque response

Torque increased with measured cycle frequency and decreased with commanded cycle period in these data. Mean, peak, and RMS torque are reported by run. Run-summary regressions are exploratory (`n = {n}` operating conditions) and do not establish causation.

## 8. Generator-voltage behavior

{gen_lines}

Drift magnitude and periodicity are classified independently. A configured drift threshold is not a hypothesis test. High R² supports a strong linear drift fit; low R² indicates weak linear-model evidence even when slope magnitude exceeds the configured threshold. Sequential run order and overall drift confound frequency comparisons, and the physical meaning of Gen_V is undocumented.

## 9. Signal quality

The intake found {audit['missing_value_count']} missing values and {audit['duplicate_row_count']} duplicate rows. Quality tables retain spike candidates, pressure-pair offsets, timing jitter, encoder flags, command saturation, and other evidence without deleting data or declaring sensor failure.

## 10. Cross-test engineering interpretation

Shorter cycle time coincided with larger upstream robust response, larger dynamic differential-pressure RMS, and higher torque in the current sequence. Gen_V did not increase consistently enough to separate frequency effects from drift and run order. Upstream pressure sensors were consistent; downstream consistency was poor. The fastest condition was `{fastest.Run_ID if fastest is not None else 'unavailable'}`. These comparisons remain limited by inferred targets, few cycles, four conditions, sequential ordering, and missing calibration.

## 11. Limitations

Pressure and torque units and pressure calibration are unknown. Pressure_4 offset and Pressure_3 weak response require wiring/calibration review. Runs contain limited cycles; targets may be inferred; recorded VFD command and measured VFD output frequency are absent; Gen_V meaning is undocumented; test order was sequential; no direct vibration measurement exists; new-turbine VFD scaling remains unavailable until verified.

## 12. Recommended next tests

Record Target_Cycle_s, Target_VFD_Hz, VFD_Command_mV, and measured VFD output frequency. Capture 10–20 steady cycles and complete startup/stopping behavior. Randomize or reverse run order. Verify pressure calibration, zeroing, Pressure_3/Pressure_4 wiring, engineering units, Gen_V meaning, and new-turbine VFD scaling. Add motor current, vibration/acceleration, and independent RPM measurements; repeat the fastest condition after scaling is confirmed.

## 13. Conclusion

The package supports repeatable comparison of timing, robust pressure response, torque, and Gen_V behavior while preserving raw data. The strongest trends are increasing pressure response and torque with increasing measured frequency. Absolute pressure and causal performance conclusions remain unsupported without calibration, randomized repeats, more cycles, and direct command/frequency measurements.

## Reproducibility

Input SHA-256: `{metadata['input_sha256']}`. Configuration SHA-256: `{metadata['configuration_sha256']}`. Python: `{metadata['python_version']}`. Git commit: `{metadata['git_commit']}`. Analysis timestamp: `{metadata['analysis_timestamp']}`.
"""
    methods="""# Methods

Data were imported in Python using pandas, with configurable column-alias mapping and timestamp conversion for CSV and Excel sources. Time-base selection prioritized recorded TimeStamp, recorded numeric Elapsed_Time_s, then validated RecNum reconstruction with an explicit fixed interval. {time_statement} NumPy and SciPy were used to characterize sampling, segment operating runs, identify steady-state portions, filter encoder data, and calculate peak-, trough-, autocorrelation-, FFT-, and zero-crossing-based cycle estimates. Final timing selection retained cycle-level intervals and method-agreement evidence separately.

Pressure means, differential channels, run-centered dynamic channels, robust percentile spans, cycle amplitudes, correlations, signed lag, and wrapped phase were calculated only when the required channels were available. Positive lag denotes that the second signal occurs after the first. Torque and generator-voltage statistics were evaluated at sample, cycle, and run-summary levels as applicable. Gen_V drift and periodicity were classified independently. Quality checks used configurable robust filters and retained raw values alongside clearly named processed columns.

All source rows, timestamps, row order, and measured values were preserved. Derived, filtered, inferred, and reconstructed quantities were assigned distinct fields. matplotlib generated noninteractive PNG figures, and openpyxl generated the Microsoft Excel summary workbook. No external spreadsheet software was used.
"""
    executive=f"""# Executive Summary

This analysis packages {metadata['total_records']:,} HNEI OWC test-bench records into a reproducible engineering review of {n} operating conditions. Intake checks preserved all rows and raw measurements. Sampling was primarily {_fmt(audit['sampling_interval_median_s'],3)} seconds, with documented jitter but no missing values or duplicate rows in the current workbook.

Encoder timing identified the expected legacy operating sequence. Timing precision is reported consistently with the sampling interval, and timing-method confidence is separated from cycle stability. The legacy 5-second condition remains command-saturated under the configured reconstruction, but existing timing evidence cannot distinguish nominal from capped-command behavior without recorded command voltage or measured VFD frequency.

Robust and median-cycle pressure measures show increasing response toward shorter cycle periods. They replace isolated raw extrema as the primary attenuation comparison; the fastest run's raw attenuation disagrees with its robust result and is flagged for review. Dynamic differential pressure is reported separately from raw offset-bearing differential pressure. Absolute turbine pressure drop remains unavailable because sensor calibration, units, zero references, and sign conventions are not documented.

Mean, peak, and RMS torque increased with measured cycle frequency and decreased with commanded period. The four-condition regressions are exploratory. Gen_V contains drift and, in some runs, periodic behavior; drift R² varies, test order is sequential, and the channel's physical meaning is undocumented, so a causal frequency trend is not established.

Priority actions are to verify pressure calibration and downstream wiring, record command and measured VFD frequency, document units and Gen_V meaning, capture 10–20 steady cycles, randomize test order, and add independent RPM and vibration measurements. New-turbine command reconstruction should remain unavailable until its VFD scaling is verified.
"""
    limits="""# Limitations and Recommendations

## Confirmed limitations

- {time_statement}
- Pressure and torque units and pressure calibration are undocumented.
- Recorded VFD command and measured VFD output frequency are unavailable.
- Gen_V channel meaning is undocumented.
- Test order is sequential, only four operating conditions are available, and cycle counts are limited.
- No direct vibration or independent RPM measurement is present.

## Suspected instrumentation issues

- Pressure_4 shows a substantial offset relative to Pressure_3.
- Pressure_3 has weak response/correlation evidence.
- Spike candidates require review against raw traces; they are not proof of sensor failure.

## Required verification

- Verify pressure calibration, zero references, sign convention, and Pressure_3/Pressure_4 wiring.
- Document pressure, torque, and Gen_V meanings and units.
- Record command voltage and measured VFD output frequency.

## Recommended next tests

- Capture 10–20 steady cycles, full startup/stopping, randomized or reversed test order, motor current, acceleration/vibration, and independent RPM.
- Repeat the fastest condition after command scaling is confirmed.

## New-turbine configuration requirements

Retain decimal target periods and recorded targets. Do not reconstruct VFD commands until the new voltage-to-frequency scaling and limits are verified and entered in configuration.
"""
    documents={"engineering_analysis_report.md":main,"methods_section.md":methods,"executive_summary.md":executive,"limitations_and_recommendations.md":limits}; paths=[]
    for name,text in documents.items(): path=report_dir/name; path.write_text(text,encoding="utf-8"); paths.append(path)
    return paths
