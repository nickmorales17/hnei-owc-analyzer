"""Descriptive, cycle-level, torque, generator, and regression statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


MEASURED_DERIVED_SIGNALS=["Pressure_1","Pressure_2","Pressure_3","Pressure_4","Upstream_Mean","Downstream_Mean","DeltaP_13","DeltaP_24","Turbine_DeltaP","Torque","Encoder","Gen_V","Target_VFD_Hz","Reconstructed_Target_VFD_Hz","VFD_Command_mV","Reconstructed_Command_mV_Capped"]

def _time(data:pd.DataFrame)->pd.Series:
    return data.Elapsed_Time_s if "Elapsed_Time_s" in data else (data.TimeStamp-data.TimeStamp.iloc[0]).dt.total_seconds()


def numeric_statistics(values: pd.Series,duration_s:float) -> dict[str,Any]:
    x=pd.to_numeric(values,errors="coerce").dropna().to_numpy(float); n=len(x)
    if not n: return {"Sample_Count":0,"Duration_s":duration_s}
    return {"Sample_Count":n,"Duration_s":duration_s,"Mean":float(np.mean(x)),"Minimum":float(np.min(x)),"Maximum":float(np.max(x)),"Peak_To_Peak":float(np.ptp(x)),"Population_Std":float(np.std(x,ddof=0)),"Sample_Std":float(np.std(x,ddof=1)) if n>=2 else np.nan,"RMS":float(np.sqrt(np.mean(x*x))),"Median":float(np.median(x)),"Percentile_5":float(np.percentile(x,5)),"Percentile_95":float(np.percentile(x,95)),"Interquartile_Range":float(np.percentile(x,75)-np.percentile(x,25)),"Mean_Absolute_Deviation":float(np.mean(np.abs(x-np.mean(x))))}


def descriptive_statistics(data:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        for section,section_data in [("entire_run",run),("startup_transient",run[run.Operating_State=="startup_transient"]),("steady_state",run[run.Operating_State=="steady_state"]),("stopping_transient",run[run.Operating_State=="stopping_transient"])]:
            duration=float(_time(section_data).max()-_time(section_data).min()) if len(section_data)>1 else 0.0
            for signal in MEASURED_DERIVED_SIGNALS:
                if signal in section_data:
                    rows.append({"Run_ID":run_id,"Section":section,"Signal":signal,"Primary_Comparison":section=="steady_state",**numeric_statistics(section_data[signal],duration)})
    return pd.DataFrame(rows)


def cycle_level_statistics(data:pd.DataFrame,config:dict[str,Any]) -> pd.DataFrame:
    rows=[]; pressure=["Pressure_1","Pressure_2","Pressure_3","Pressure_4","Upstream_Mean","Downstream_Mean","Turbine_DeltaP","Turbine_DeltaP_Dynamic","Turbine_DeltaP_Cycle_Dynamic"]
    steady=data[(data.Run_ID.notna())&data.Run_ID.astype(str).ne("")&(data.Final_Cycle_Number.notna())&(data.Is_Steady_State)]
    for (run_id,cycle),group in steady.groupby(["Run_ID","Final_Cycle_Number"]):
        for signal in pressure:
            if signal not in group: continue
            x=group[signal].dropna().to_numpy(float)
            rows.append({"Run_ID":run_id,"Cycle_Number":int(cycle),"Signal":signal,"Category":"pressure","Mean":np.mean(x),"Minimum":np.min(x),"Maximum":np.max(x),"Positive_Peak":np.max(x),"Negative_Peak":np.min(x),"Peak_To_Peak":np.ptp(x),"RMS":np.sqrt(np.mean(x*x)),"Sample_Std":np.std(x,ddof=1) if len(x)>1 else np.nan,"Linear_Drift_Per_s":np.nan,"Cycle_Quality_Flag":""})
        for signal,category in [("Torque","torque"),("Gen_V","generator_voltage")]:
            if signal not in group: continue
            x=group[signal].dropna().to_numpy(float); t=(_time(group)-_time(group).iloc[0]).to_numpy(); slope=float(linregress(t,x).slope) if len(x)>=3 and np.ptp(t)>0 else np.nan
            rows.append({"Run_ID":run_id,"Cycle_Number":int(cycle),"Signal":signal,"Category":category,"Mean":np.mean(x),"Minimum":np.min(x),"Maximum":np.max(x),"Positive_Peak":np.max(x),"Negative_Peak":np.min(x),"Peak_To_Peak":np.ptp(x),"RMS":np.sqrt(np.mean(x*x)),"Sample_Std":np.std(x,ddof=1) if len(x)>1 else np.nan,"Linear_Drift_Per_s":slope,"Cycle_Quality_Flag":""})
    return pd.DataFrame(rows)


def regression_metrics(x:pd.Series,y:pd.Series,data_level:str,relationship:str,minimum:int=3) -> dict[str,Any]:
    pair=pd.DataFrame({"x":pd.to_numeric(x,errors="coerce"),"y":pd.to_numeric(y,errors="coerce")}).dropna(); n=len(pair)
    base={"Relationship":relationship,"Data_Level":data_level,"Observation_Count":n}
    if n<minimum or pair.x.std()==0 or pair.y.std()==0: return {**base,"Pearson":np.nan,"Spearman":np.nan,"Slope":np.nan,"Intercept":np.nan,"R_Squared":np.nan,"Caution":"insufficient observations"}
    fit=linregress(pair.x,pair.y); caution="Only four test conditions; run-summary R-squared is exploratory, not conclusive." if data_level=="run summary" and n<=4 else ""
    return {**base,"Pearson":float(pearsonr(pair.x,pair.y).statistic),"Spearman":float(spearmanr(pair.x,pair.y).statistic),"Slope":fit.slope,"Intercept":fit.intercept,"R_Squared":fit.rvalue**2,"Caution":caution}


def torque_summary(data:pd.DataFrame,cycle_stats:pd.DataFrame,run_periods:dict[str,float]) -> pd.DataFrame:
    rows=[]
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        if "Torque" not in run: continue
        steady=run[run.Is_Steady_State]; x=steady.Torque.dropna(); cycles=cycle_stats[(cycle_stats.Run_ID==run_id)&(cycle_stats.Signal=="Torque")]
        row={"Run_ID":run_id,"Measured_Cycle_s":run_periods.get(str(run_id),np.nan),"Measured_Cycle_Frequency_Hz":1/run_periods[str(run_id)] if str(run_id) in run_periods else np.nan,"Mean_Torque":x.mean(),"Peak_Torque":x.max(),"Minimum_Torque":x.min(),"RMS_Torque":np.sqrt(np.mean(x*x)),"Positive_Torque_Fraction":float((x>0).mean()),"Negative_Torque_Fraction":float((x<0).mean()),"Cycle_To_Cycle_Peak_Std":cycles.Maximum.std(ddof=1) if len(cycles)>1 else np.nan,"Cycle_To_Cycle_RMS_Std":cycles.RMS.std(ddof=1) if len(cycles)>1 else np.nan}
        for signal in ["Turbine_DeltaP","Upstream_Mean","Downstream_Mean","Encoder"]:
            if signal in steady: row[f"Torque_vs_{signal}_Pearson"]=regression_metrics(steady[signal],steady.Torque,"sample",f"Torque versus {signal}")["Pearson"]
        rows.append(row)
    return pd.DataFrame(rows)


def generator_summary(data:pd.DataFrame,run_periods:dict[str,float],config:dict[str,Any]|None=None) -> pd.DataFrame:
    rows=[]
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        if "Gen_V" not in run: continue
        x=run.Gen_V.dropna(); t=_time(run).loc[x.index]-_time(run).loc[x.index].iloc[0]; fit=linregress(t,x) if len(x)>=3 and np.ptp(t)>0 else None; period=run_periods.get(str(run_id),np.nan)
        autocorr_strength=np.nan; dominant_amplitude=np.nan; dominant_period=np.nan; periodicity_confidence=0.0
        if len(x)>3 and np.std(x)>0:
            centered=x.to_numpy()-x.mean(); lag=int(round(period/np.median(np.diff(t)))) if np.isfinite(period) else 0
            if 0<lag<len(centered): autocorr_strength=float(np.corrcoef(centered[:-lag],centered[lag:])[0,1])
            detrended=centered-(fit.slope*t.to_numpy()+fit.intercept-x.mean() if fit else 0); frequencies=np.fft.rfftfreq(len(detrended),float(np.median(np.diff(t)))); amplitudes=2*np.abs(np.fft.rfft(detrended))/len(detrended)
            mask=(frequencies>=.1)&(frequencies<=1/1.5)
            if mask.any():
                candidates=np.flatnonzero(mask); peak=candidates[int(np.argmax(amplitudes[mask]))]; dominant_amplitude=float(amplitudes[peak]); dominant_period=float(1/frequencies[peak]); periodicity_confidence=float(min(1,dominant_amplitude/(2*np.std(detrended)+1e-12)))
        slope=fit.slope if fit else np.nan; r2=fit.rvalue**2 if fit else np.nan; total=slope*(t.iloc[-1]-t.iloc[0]) if fit else np.nan
        settings=(config or {}).get("stage5",{}); constant=x.std(ddof=0)<float(settings.get("near_constant_std",1e-6)); drift_significant=np.isfinite(total) and abs(total)>max(float(settings.get("genv_minimum_total_drift",.01)),float(settings.get("genv_drift_std_fraction",.25))*x.std(ddof=0)); periodic=np.isfinite(autocorr_strength) and autocorr_strength>float(settings.get("genv_periodicity_correlation",.5)) and periodicity_confidence>=float(settings.get("genv_periodicity_confidence",.15))
        drift_class="nearly_constant" if constant else "significant_drift" if drift_significant else "no_significant_drift"; periodic_class="periodic" if periodic else "nonperiodic"; drift_confidence=float(min(1,abs(total)/(x.std(ddof=0)+1e-12))) if np.isfinite(total) else 0
        combined="nearly_constant" if constant else "periodic_with_drift" if periodic and drift_significant else "periodic_without_significant_drift" if periodic else "drifting_nonperiodic" if drift_significant else "unknown"
        rows.append({"Run_ID":run_id,**numeric_statistics(x,float(t.iloc[-1]-t.iloc[0]) if len(t)>1 else 0),"Linear_Drift_Per_s":slope,"Total_Fitted_Drift":total,"Drift_R_Squared":r2,"Dominant_Periodic_Amplitude":dominant_amplitude,"Dominant_Period_s":dominant_period,"Periodicity_Confidence":periodicity_confidence,"Drift_Confidence":drift_confidence,"Cycle_Autocorrelation":autocorr_strength,"Drift_Classification":drift_class,"Periodicity_Classification":periodic_class,"Combined_GenV_Classification":combined,"Behavior_Classification":combined,"Physical_Channel_Warning":"Gen_V physical channel meaning is undocumented.","Run_Order_Drift_Warning":"Sequential run order and overall drift confound frequency comparisons."})
    return pd.DataFrame(rows)


def correlation_regression_summary(data:pd.DataFrame,cycle_stats:pd.DataFrame,torque:pd.DataFrame,gen:pd.DataFrame,pressure:pd.DataFrame,minimum:int=3) -> pd.DataFrame:
    rows=[]
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        steady=run[run.Is_Steady_State]
        for x,y in [("Turbine_DeltaP","Torque"),("Upstream_Mean","Torque"),("Downstream_Mean","Torque"),("Torque","Gen_V"),("Turbine_DeltaP","Gen_V")]:
            if x in steady and y in steady: rows.append({"Run_ID":run_id,**regression_metrics(steady[x],steady[y],"sample",f"{y} versus {x}",minimum)})
        torque_cycles=cycle_stats[(cycle_stats.Run_ID==run_id)&(cycle_stats.Signal=="Torque")]
        dp_cycles=cycle_stats[(cycle_stats.Run_ID==run_id)&(cycle_stats.Signal=="Turbine_DeltaP")]
        if not torque_cycles.empty and not dp_cycles.empty:
            merged=torque_cycles.merge(dp_cycles,on=["Run_ID","Cycle_Number"],suffixes=("_Torque","_DeltaP")); rows.append({"Run_ID":run_id,**regression_metrics(merged.RMS_DeltaP,merged.RMS_Torque,"cycle","Torque RMS versus Turbine_DeltaP RMS",minimum)})
    if not torque.empty:
        for metric in ["Mean_Torque","Peak_Torque","RMS_Torque"]: rows.append({"Run_ID":"all_runs",**regression_metrics(torque.Measured_Cycle_Frequency_Hz,torque[metric],"run summary",f"{metric} versus measured cycle frequency",minimum)})
    if not gen.empty and not torque.empty:
        merged=gen.merge(torque,on="Run_ID"); rows.append({"Run_ID":"all_runs",**regression_metrics(merged.Measured_Cycle_Frequency_Hz,merged.Mean,"run summary","Mean Gen_V versus measured cycle frequency",minimum)})
    return pd.DataFrame(rows)
