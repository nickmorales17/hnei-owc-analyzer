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
    return data["Elapsed_Time_s"] - data["Elapsed_Time_s"].iloc[0]


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


def create_final_graph_package(data:pd.DataFrame,blocks:list[RunBlock],tables:dict[str,pd.DataFrame],output_dir:Path,config:dict[str,Any])->list[Path]:
    """Create polished Stage 6 time-history, comparison, correlation, and quality PNGs."""
    dpi=int(config.get("reporting",{}).get("plot_dpi",180)); created=[]
    dirs={name:output_dir/"graphs"/name for name in ["time_histories","comparison","correlation","quality"]}
    for directory in dirs.values(): directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name,category):
        path=dirs[category]/name; fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig); created.append(path)
    def shade(ax,run):
        x=run.Elapsed_Time_s-run.Elapsed_Time_s.iloc[0]
        for state,color in STATE_COLORS.items():
            rows=run.index[run.Operating_State==state] if "Operating_State" in run else []
            if len(rows): ax.axvspan(x.loc[rows[0]],x.loc[rows[-1]],color=color,alpha=.10)
    for block in blocks:
        run=data.loc[block.start_row:block.end_row]; x=run.Elapsed_Time_s-run.Elapsed_Time_s.iloc[0]; prefix=block.run_id
        specs=[(["Pressure_1","Pressure_2","Pressure_3","Pressure_4"],"pressure_channels","Pressure channels"),(["Upstream_Mean","Downstream_Mean"],"pressure_means","Upstream/downstream means"),(["Turbine_DeltaP_Dynamic"],"dynamic_delta_p","Dynamic turbine differential pressure"),(["Torque","Torque_Filtered"],"torque","Torque raw and filtered"),(["Encoder","Encoder_Smoothed"],"encoder","Encoder raw and processed"),(["Gen_V"],"generator_voltage","Gen_V and fitted drift")]
        for columns,suffix,title in specs:
            available=[c for c in columns if c in run]
            if not available: continue
            fig,ax=plt.subplots(figsize=(10,4));
            for c in available: ax.plot(x,run[c],lw=.7,label=c)
            if suffix=="encoder" and "Final_Cycle_Number" in run:
                changes=run.Final_Cycle_Number.ne(run.Final_Cycle_Number.shift())&run.Final_Cycle_Number.notna(); ax.scatter(x[changes],run.loc[changes,"Encoder_Smoothed"],s=12,label="Cycle marker")
            if suffix=="generator_voltage" and len(run)>=3:
                fit=np.polyfit(x,run.Gen_V,1); ax.plot(x,np.polyval(fit,x),ls="--",label="Linear drift fit")
            shade(ax,run); ax.set(title=f"{prefix} — {title}",xlabel="Elapsed run time (s)",ylabel="Recorded/configured units"); ax.grid(alpha=.25); ax.legend(fontsize=7); save(fig,f"{prefix}_{suffix}.png","time_histories")
        fig,ax=plt.subplots(figsize=(10,2.8)); codes={s:i for i,s in enumerate(STATE_COLORS)}; ax.scatter(x,run.Operating_State.map(codes),c=run.Operating_State.map(STATE_COLORS),s=4); ax.set_yticks(list(codes.values()),list(codes)); ax.set(title=f"{prefix} — operating state",xlabel="Elapsed run time (s)"); ax.grid(alpha=.2); save(fig,f"{prefix}_operating_state.png","time_histories")
        vfd=[c for c in ["Reconstructed_Target_VFD_Hz","VFD_Command_mV","Reconstructed_Command_mV_Capped"] if c in run and run[c].notna().any()]
        if vfd:
            fig,ax=plt.subplots(figsize=(10,4));
            for c in vfd: ax.plot(x,run[c],label=c)
            shade(ax,run); ax.set(title=f"{prefix} — VFD target/command channels",xlabel="Elapsed run time (s)",ylabel="Configured units"); ax.grid(alpha=.25); ax.legend(fontsize=7); save(fig,f"{prefix}_vfd_channels.png","time_histories")
    cross=tables["cross_test"].copy(); freq=cross.Final_Selected_Frequency_Hz; target=cross.Target_Cycle_s; labels=cross.Run_ID
    comparison=[("Final_Selected_Period_s",target,"Measured cycle versus target","Target cycle (s)","Measured cycle (s)"),("Sample_CV_Percent",target,"Cycle-time sample CV","Target cycle (s)","Sample CV (%)"),("Mean_Torque",freq,"Mean torque versus measured frequency","Measured frequency (Hz)","Mean torque (unknown units)"),("Peak_Torque",freq,"Peak torque versus measured frequency","Measured frequency (Hz)","Peak torque (unknown units)"),("RMS_Torque",freq,"RMS torque versus measured frequency","Measured frequency (Hz)","RMS torque (unknown units)"),("Robust_Upstream_Pressure_Span",freq,"Robust upstream response","Measured frequency (Hz)","P5–P95 span (unknown units)"),("Robust_Downstream_Pressure_Span",freq,"Robust downstream response","Measured frequency (Hz)","P5–P95 span (unknown units)"),("Dynamic_DeltaP_RMS",freq,"Dynamic differential-pressure RMS","Measured frequency (Hz)","Dynamic RMS (unknown units)"),("Robust_Attenuation_Ratio",freq,"Robust attenuation ratio","Measured frequency (Hz)","Downstream/upstream ratio"),("Median_Cycle_Attenuation_Ratio",freq,"Median-cycle attenuation","Measured frequency (Hz)","Downstream/upstream ratio"),("Mean_Gen_V",freq,"Mean Gen_V versus frequency","Measured frequency (Hz)","Mean Gen_V (V)"),("Gen_V_Drift_Slope",np.arange(1,len(cross)+1),"Gen_V drift versus run order","Run order","Drift slope (V/s)")]
    cross["Cycle_Error_s"]=cross.Final_Selected_Period_s-cross.Target_Cycle_s
    comparison.insert(1,("Cycle_Error_s",target,"Cycle error versus target","Target cycle (s)","Cycle error (s)"))
    for field,xv,title,xlabel,ylabel in comparison:
        if field not in cross: continue
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(xv,cross[field],"o-")
        for xval,yval,label in zip(xv,cross[field],labels): ax.annotate(str(label),(xval,yval),fontsize=7)
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25); save(fig,f"{field.lower()}.png","comparison")
    vfd=tables["vfd"]
    for fields,title,name,ylabel in [(["Desired_Target_Frequency_Hz","Command_Equivalent_Frequency_Hz","Final_Measured_VFD_Equivalent_Frequency_Hz"],"Desired, equivalent, and measured frequency","vfd_frequency_comparison","VFD-equivalent frequency (Hz)"),(["Reconstructed_Command_mV_Uncapped","Reconstructed_Command_mV_Capped"],"Uncapped and capped VFD command","vfd_command_comparison","Command (mV)")]:
        available=[c for c in fields if c in vfd and vfd[c].notna().any()]
        if not available: continue
        fig,ax=plt.subplots(figsize=(8,4)); vfd.set_index("Run_ID")[available].plot(kind="bar",ax=ax); ax.set(title=title,ylabel=ylabel); ax.grid(axis="y",alpha=.25); save(fig,f"{name}.png","comparison")
    phase=tables["pressure_phase"]; reliable=phase[(phase.Reliability=="reliable")&(phase.Data_Version=="raw")]
    if not reliable.empty:
        merged=reliable.merge(cross[["Run_ID","Final_Selected_Frequency_Hz"]],on="Run_ID"); fig,ax=plt.subplots(figsize=(7,4));
        for rel,group in merged.groupby(merged.Signal_1+"→"+merged.Signal_2): ax.scatter(group.Final_Selected_Frequency_Hz,group.Wrapped_Phase_Degrees,label=rel)
        ax.set(title="Reliable pressure phase versus measured frequency",xlabel="Measured frequency (Hz)",ylabel="Wrapped phase (degrees)"); ax.legend(fontsize=6); ax.grid(alpha=.25); save(fig,"pressure_phase_vs_frequency.png","comparison")
    correlation_specs=[("Turbine_DeltaP_Dynamic","Torque","Torque versus dynamic ΔP"),("Upstream_Dynamic","Torque","Torque versus upstream dynamic pressure"),("Torque","Gen_V","Gen_V versus torque"),("Turbine_DeltaP_Dynamic","Gen_V","Gen_V versus dynamic ΔP"),("Upstream_Dynamic","Downstream_Dynamic","Upstream versus downstream dynamic pressure")]
    steady=data[data.Is_Steady_State]
    for xcol,ycol,title in correlation_specs:
        if xcol not in steady or ycol not in steady: continue
        fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(steady[xcol],steady[ycol],s=4,alpha=.3); ax.set(title=f"{title} — sample level",xlabel=f"{xcol} (configured units)",ylabel=f"{ycol} (configured units)"); ax.grid(alpha=.25); save(fig,f"{xcol.lower()}_vs_{ycol.lower()}.png","correlation")
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(freq,cross.Mean_Torque); ax.set(title=f"Mean torque versus frequency — run summary, n = {len(cross)}",xlabel="Measured frequency (Hz)",ylabel="Mean torque (unknown units)"); ax.grid(alpha=.25); save(fig,"torque_vs_frequency_run_summary.png","correlation")
    torque_phase=tables["torque_phase"]
    if not torque_phase.empty:
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(torque_phase.Run_ID+"\n"+torque_phase.Relationship,torque_phase.Phase_Deg); ax.set(title="Torque phase relationships",ylabel="Wrapped phase (degrees)"); ax.tick_params(axis="x",rotation=45,labelsize=6); ax.grid(axis="y",alpha=.25); save(fig,"torque_phase_relationships.png","correlation")
    from .pressure_analysis import lagged_cross_correlation
    from .correlation_analysis import phase_relationship
    sampling=float(data.Elapsed_Time_s.diff().median())
    for run_id,run in steady[steady.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        period=float(cross.loc[cross.Run_ID==run_id,"Final_Selected_Period_s"].iloc[0]); max_lag=max(1,int(round(.5*period/sampling)))
        if {"Upstream_Dynamic","Downstream_Dynamic"}.issubset(run.columns):
            result=lagged_cross_correlation(run.Upstream_Dynamic,run.Downstream_Dynamic,max_lag); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(result["lags"]*sampling,result["correlations"]); ax.axvline(result["lag_samples"]*sampling,ls="--",color="red",label="Selected signed lag"); ax.set(title=f"{run_id} upstream/downstream cross-correlation — sample level",xlabel="Lag (s); positive means downstream occurs after upstream",ylabel="Correlation"); ax.legend(); ax.grid(alpha=.25); save(fig,f"{run_id}_pressure_cross_correlation.png","correlation")
        if {"Turbine_DeltaP_Dynamic","Torque"}.issubset(run.columns):
            phase=phase_relationship(run.Turbine_DeltaP_Dynamic,run.Torque,period,sampling); fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["Torque relative to\ndynamic ΔP"],[phase["Phase_Deg"]]); ax.set(title=f"{run_id} torque/dynamic ΔP phase — sample level",ylabel="Wrapped phase (degrees)"); ax.grid(axis="y",alpha=.25); save(fig,f"{run_id}_torque_dynamic_delta_p_phase.png","correlation")
    q=tables["quality_findings"]
    if not q.empty:
        counts=q.groupby("Signal").size(); fig,ax=plt.subplots(figsize=(7,4)); counts.plot(kind="bar",ax=ax); ax.set(title="Signal-quality findings by channel",ylabel="Finding count"); save(fig,"quality_flags_by_channel.png","quality")
        spikes=q[q.Check=="isolated_spikes"].groupby(["Run_ID","Signal"]).size().unstack(fill_value=0); fig,ax=plt.subplots(figsize=(8,4)); spikes.plot(kind="bar",ax=ax); ax.set(title="Spike findings by run and channel",ylabel="Finding records"); save(fig,"spikes_by_run_channel.png","quality")
    response=tables["pressure_response"]; fig,ax=plt.subplots(figsize=(8,4)); response.set_index("Run_ID")[["Raw_Attenuation_Ratio","Robust_Attenuation_Ratio","Median_Cycle_Attenuation_Ratio"]].plot(kind="bar",ax=ax); ax.set(title="Raw, robust, and median-cycle attenuation",ylabel="Attenuation ratio"); save(fig,"attenuation_comparison.png","quality")
    consistency=tables["pressure_consistency"]
    if not consistency.empty:
        offsets=tables["pressure_pairs"][(tables["pressure_pairs"].Pair=="downstream_pair")&(tables["pressure_pairs"].Data_Version=="raw")]; fig,ax=plt.subplots(figsize=(7,4)); ax.bar(offsets.Run_ID,offsets.Mean_Offset); ax.set(title="Downstream sensor offset by run",ylabel="Pressure_4 − Pressure_3 (unknown units)"); save(fig,"downstream_sensor_offset.png","quality")
    gen=tables["generator"]; fig,ax=plt.subplots(figsize=(8,4)); ax.bar(gen.Run_ID,gen.Linear_Drift_Per_s,label="Drift slope"); ax2=ax.twinx(); ax2.plot(gen.Run_ID,gen.Drift_R_Squared,"o--",color="black",label="R²"); ax.set(title="Gen_V drift slope and fit",ylabel="Slope (V/s)"); ax2.set_ylabel("R²"); save(fig,"gen_v_drift_quality.png","quality")
    intervals=data.Elapsed_Time_s.diff().dropna(); fig,ax=plt.subplots(figsize=(7,4)); ax.hist(intervals,bins=30); ax.set(title="Sample interval distribution" if data.Time_Source.iloc[0] != "reconstructed_from_record_number" else "Record-step interval distribution (assumed time base)",xlabel="Interval (s)",ylabel="Count"); save(fig,"sampling_interval_histogram.png","quality")
    cycles=tables["encoder_intervals"]; fig,ax=plt.subplots(figsize=(8,4));
    for run_id,group in cycles.groupby("Run_ID"): ax.plot(group.Cycle_Number,group.Measured_Cycle_s,"o-",label=run_id)
    ax.set(title="Cycle-period variation by run",xlabel="Cycle number",ylabel="Measured cycle (s)"); ax.legend(); ax.grid(alpha=.25); save(fig,"cycle_period_variation.png","quality")
    return created
