"""Stage 3 diagnostic plotting only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .test_segmentation import RunBlock


STATE_COLORS = {"idle": "0.8", "startup_transient": "#f4a261", "steady_state": "#2a9d8f", "stopping_transient": "#e76f51", "unclassified": "#9b5de5"}


def _elapsed(data: pd.DataFrame) -> pd.Series:
    return (data["TimeStamp"] - data["TimeStamp"].iloc[0]).dt.total_seconds()


def _shade_runs(axis: plt.Axes, data: pd.DataFrame, blocks: list[RunBlock]) -> None:
    elapsed = _elapsed(data)
    for block in blocks:
        label = f"{block.run_id}: {block.provisional_target_cycle_s or 'unclassified'} s ({block.target_source})"
        axis.axvspan(elapsed.iloc[block.start_row], elapsed.iloc[block.end_row], alpha=0.12, label=label)


def create_stage3_diagnostics(data: pd.DataFrame, blocks: list[RunBlock], results: dict[str, tuple[pd.DataFrame, dict[str, Any], np.ndarray]], output_dir: Path, config: dict[str, Any]) -> list[Path]:
    graph_dir = output_dir / "graphs" / "stage3_diagnostics"
    graph_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("plotting", {}).get("dpi", 180))
    created: list[Path] = []
    elapsed = _elapsed(data)
    for signal, filename, ylabel in [("Encoder", "encoder_run_boundaries.png", "Encoder (recorded units)"), ("Pressure_1", "pressure_1_run_boundaries.png", "Pressure_1 (configured units)")]:
        if signal not in data:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(elapsed, data[signal], linewidth=0.7, color="black", label=signal)
        _shade_runs(ax, data, blocks)
        ax.set(title=f"Stage 3 inferred run boundaries — {signal}", xlabel="Elapsed time (s)", ylabel=ylabel)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
        path = graph_dir / filename; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    codes = {state: index for index, state in enumerate(STATE_COLORS)}
    ax.scatter(elapsed, data["Operating_State"].map(codes), c=data["Operating_State"].map(STATE_COLORS), s=3)
    ax.set_yticks(list(codes.values()), list(codes.keys())); ax.set(title="Stage 3 operating-state classification", xlabel="Elapsed time (s)", ylabel="State"); ax.grid(alpha=0.25)
    path = graph_dir / "operating_state_classification.png"; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)

    for block in blocks:
        segment = data.loc[block.start_row:block.end_row]
        local_elapsed = elapsed.loc[segment.index]
        cycles, summary, peaks = results[block.run_id]
        for signal, suffix in [("Encoder", "encoder_peaks"), ("Pressure_1", "pressure_states")]:
            if signal not in segment:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(local_elapsed, segment[signal], linewidth=0.8, color="black", label=signal)
            if signal == "Encoder" and len(peaks):
                ax.scatter(elapsed.iloc[peaks], data.loc[peaks, signal], s=18, color="red", label="Detected cycle peaks")
            for state, color in STATE_COLORS.items():
                state_rows = segment.index[segment["Operating_State"] == state]
                if len(state_rows):
                    ax.axvspan(elapsed.loc[state_rows[0]], elapsed.loc[state_rows[-1]], color=color, alpha=0.15, label=state)
            target = block.provisional_target_cycle_s if block.provisional_target_cycle_s is not None else "unclassified"
            ax.set(title=f"{block.run_id} — inferred target {target} s — {signal}", xlabel="Elapsed time (s)", ylabel=f"{signal} (recorded units)")
            ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=3)
            path = graph_dir / f"{block.run_id}_{suffix}.png"; fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig); created.append(path)
    return created


def create_stage4_diagnostics(data: pd.DataFrame, blocks: list[RunBlock], cycle_tables: dict[str, pd.DataFrame], method_table: pd.DataFrame, summary_table: pd.DataFrame, vfd_table: pd.DataFrame, event_indices: dict[str, dict[str, np.ndarray]], output_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Create only the plots needed to verify Stage 4 timing and commands."""
    graph_dir = output_dir / "graphs" / "stage4_diagnostics"; graph_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("plotting", {}).get("dpi", 180)); elapsed = _elapsed(data); created: list[Path] = []
    for block in blocks:
        segment = data.loc[block.start_row:block.end_row]
        fig, ax = plt.subplots(figsize=(10,4)); ax.plot(elapsed.loc[segment.index],segment["Encoder"],lw=.55,label="Encoder raw (recorded)"); ax.plot(elapsed.loc[segment.index],segment["Encoder_Median_Filtered"],lw=.8,label="Median filtered"); ax.plot(elapsed.loc[segment.index],segment["Encoder_Smoothed"],lw=1.1,label="Savitzky-Golay timing signal"); ax.set(title=f"{block.run_id} raw and processed encoder",xlabel="Elapsed time (s)",ylabel="Encoder (recorded units)"); ax.grid(alpha=.25); ax.legend(fontsize=7); path=graph_dir/f"{block.run_id}_raw_filtered_encoder.png"; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
        steady = segment[segment["Is_Steady_State"]]
        if len(steady):
            events=event_indices.get(block.run_id,{}); fig,ax=plt.subplots(figsize=(10,4)); ax.plot(elapsed.loc[steady.index],steady["Encoder_Smoothed"],label="Filtered encoder")
            for key,color,marker in [("peaks","red","^"),("troughs","blue","v")]:
                local=events.get(key,np.array([],dtype=int)); global_rows=steady.index.to_numpy()[local] if len(local) else np.array([],dtype=int); ax.scatter(elapsed.loc[global_rows],steady.loc[global_rows,"Encoder_Smoothed"],c=color,marker=marker,s=28,label=f"Detected {key}")
            ax.set(title=f"{block.run_id} steady-state peaks and troughs",xlabel="Elapsed time (s)",ylabel="Filtered encoder (recorded units)"); ax.grid(alpha=.25); ax.legend(); path=graph_dir/f"{block.run_id}_peaks_troughs.png"; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
        cycles=cycle_tables.get(block.run_id,pd.DataFrame()); fig,ax=plt.subplots(figsize=(8,4))
        if not cycles.empty:
            colors=np.where(cycles.Interval_Outlier_Flag,"red","#2a9d8f"); ax.scatter(cycles.Cycle_Number,cycles.Measured_Cycle_s,c=colors,label="Cycle interval"); ax.axhline(cycles.Measured_Cycle_s.median(),ls="--",color="black",label="Run median")
        summary_row = summary_table.loc[summary_table.Run_ID == block.run_id]
        cv_label = ""
        if not summary_row.empty:
            cv_label = f" — valid sample CV {summary_row.Valid_Sample_CV_Ratio.iloc[0]:.6f} ({summary_row.Valid_Sample_CV_Percent.iloc[0]:.3f}%)"
        ax.set(title=f"{block.run_id} measured cycle intervals{cv_label}",xlabel="Cycle number",ylabel="Measured cycle time (s)"); ax.grid(alpha=.25); ax.legend(); path=graph_dir/f"{block.run_id}_cycle_intervals.png"; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
    pivot=method_table.pivot(index="Run_ID",columns="method",values="period_s"); fig,ax=plt.subplots(figsize=(9,4)); pivot.plot(kind="bar",ax=ax); ax.set(title="Final encoder period estimates by method",xlabel="Run ID",ylabel="Period (s)"); ax.grid(axis="y",alpha=.25); path=graph_dir/"method_period_comparison.png"; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
    merged=vfd_table.merge(summary_table[["Run_ID","Final_Selected_Period_s"]],on="Run_ID",how="left"); x=np.arange(len(merged)); labels=merged.Run_ID
    for columns,title,ylabel,filename in [(["Nominal_Target_Cycle_s","Command_Equivalent_Cycle_s","Final_Selected_Period_s"],"Final selected period versus nominal and capped-command cycle","Cycle time (s)","cycle_target_comparison.png"),(["Desired_Target_Frequency_Hz","Command_Equivalent_Frequency_Hz","Final_Measured_VFD_Equivalent_Frequency_Hz"],"Desired, equivalent, and measured VFD-equivalent frequency","VFD-equivalent frequency (Hz)","frequency_comparison.png"),(["Reconstructed_Command_mV_Uncapped","Reconstructed_Command_mV_Capped"],"Reconstructed uncapped and capped command","Command (mV)","command_voltage_comparison.png")]:
        fig,ax=plt.subplots(figsize=(9,4)); width=.8/len(columns)
        for i,column in enumerate(columns): ax.bar(x+(i-(len(columns)-1)/2)*width,merged[column],width,label=column)
        ax.set_xticks(x,labels); ax.set(title=title,xlabel="Run ID",ylabel=ylabel); ax.grid(axis="y",alpha=.25); ax.legend(fontsize=7); path=graph_dir/filename; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
    return created


def create_stage5_diagnostics(data: pd.DataFrame, blocks: list[RunBlock], pressure_pairs: pd.DataFrame, pressure_response: pd.DataFrame, torque: pd.DataFrame, generator: pd.DataFrame, quality_counts: pd.DataFrame, output_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Create verification plots for Stage 5 without changing source signals."""
    graph_dir=output_dir/"graphs"/"stage5_diagnostics"; graph_dir.mkdir(parents=True,exist_ok=True)
    dpi=int(config.get("plotting",{}).get("dpi",180)); elapsed=_elapsed(data); created=[]
    def save(fig:plt.Figure,name:str)->None:
        path=graph_dir/name; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
    for block in blocks:
        run=data.loc[block.start_row:block.end_row]; x=elapsed.loc[run.index]
        for columns,name,title in [(["Pressure_1","Pressure_2","Pressure_3","Pressure_4"],"pressures","Recorded pressure channels"),(["Upstream_Mean","Downstream_Mean"],"upstream_downstream","Upstream and downstream means"),(["Turbine_DeltaP"],"turbine_delta_p","Turbine differential pressure"),(["Torque"],"torque","Torque response"),(["Gen_V"],"generator_voltage","Generator-voltage response")]:
            available=[c for c in columns if c in run]
            if not available: continue
            fig,ax=plt.subplots(figsize=(10,4))
            for c in available: ax.plot(x,run[c],lw=.7,label=c)
            ax.set(title=f"{block.run_id} — {title}",xlabel="Elapsed time (s)",ylabel="Recorded/configured units"); ax.grid(alpha=.25); ax.legend(fontsize=7); save(fig,f"{block.run_id}_{name}.png")
        available=[c for c in ["Pressure_1","Pressure_2","Pressure_3","Pressure_4"] if f"{c}_Spike_Flag" in run]
        if available:
            fig,ax=plt.subplots(figsize=(10,4))
            for c in available:
                ax.plot(x,run[c],lw=.5,label=c); flagged=run[f"{c}_Spike_Flag"]
                ax.scatter(x[flagged],run.loc[flagged,c],s=14,marker="x")
            ax.set(title=f"{block.run_id} — pressure spike candidates (raw retained)",xlabel="Elapsed time (s)",ylabel="Pressure (configured units)"); ax.grid(alpha=.25); ax.legend(fontsize=7); save(fig,f"{block.run_id}_pressure_quality.png")
    if not pressure_pairs.empty:
        subset=pressure_pairs[(pressure_pairs.Pair.isin(["upstream_pair","downstream_pair","upstream_to_downstream_mean"]))&(pressure_pairs.Data_Version=="raw")]
        labels=subset.Run_ID.astype(str)+" — "+subset.Pair.str.replace("upstream_to_downstream_mean","upstream/downstream mean").str.replace("_pair"," pair")
        fig,ax=plt.subplots(figsize=(9,6)); ax.barh(labels,subset.Wrapped_Phase_Degrees); ax.set(title="Pressure-pair phase difference",xlabel="Phase (degrees; wrapped ±180°)",ylabel="Run and relationship"); ax.grid(axis="x",alpha=.25); save(fig,"pressure_pair_phase.png")
        fig,ax=plt.subplots(figsize=(9,6)); ax.barh(labels,subset.Peak_To_Peak_Amplitude_Ratio); ax.axvline(1,color="black",ls="--"); ax.set(title="Pressure-pair peak-to-peak amplitude ratios",xlabel="Second / first amplitude ratio",ylabel="Run and relationship"); ax.grid(axis="x",alpha=.25); save(fig,"pressure_pair_amplitude_ratios.png")
    if not torque.empty:
        for metric in ["Mean_Torque","Peak_Torque","RMS_Torque"]:
            fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(torque.Measured_Cycle_Frequency_Hz,torque[metric]);
            for _,r in torque.iterrows(): ax.annotate(str(r.Run_ID),(r.Measured_Cycle_Frequency_Hz,r[metric]),fontsize=7)
            ax.set(title=f"{metric} versus measured cycle frequency",xlabel="Measured cycle frequency (Hz)",ylabel=f"{metric} (recorded units)"); ax.grid(alpha=.25); save(fig,f"{metric.lower()}_vs_frequency.png")
    if not generator.empty and not torque.empty:
        merged=generator.merge(torque,on="Run_ID")
        for xcol,xlabel,name in [("Measured_Cycle_Frequency_Hz","Measured cycle frequency (Hz)","gen_v_vs_frequency"),("Mean_Torque","Mean torque (recorded units)","gen_v_vs_torque")]:
            fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(merged[xcol],merged.Mean)
            for _,r in merged.iterrows(): ax.annotate(str(r.Run_ID),(r[xcol],r.Mean),fontsize=7)
            ax.set(title="Mean Gen_V exploratory comparison",xlabel=xlabel,ylabel="Mean Gen_V (V)"); ax.grid(alpha=.25); save(fig,f"{name}.png")
    if not quality_counts.empty:
        q=quality_counts.groupby("Check").Finding_Count.sum().sort_values(); fig,ax=plt.subplots(figsize=(8,4)); q.plot(kind="barh",ax=ax); ax.set(title="Stage 5 quality findings",xlabel="Finding count",ylabel="Check"); ax.grid(axis="x",alpha=.25); save(fig,"quality_finding_counts.png")
    return created
