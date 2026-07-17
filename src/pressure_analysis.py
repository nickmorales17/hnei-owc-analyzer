"""Derived pressure channels, pair relationships, and physical consistency checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PRESSURE_PAIRS = [
    ("Pressure_1", "Pressure_2", "upstream_pair"),
    ("Pressure_3", "Pressure_4", "downstream_pair"),
    ("Pressure_1", "Pressure_3", "upstream_to_downstream_13"),
    ("Pressure_2", "Pressure_4", "upstream_to_downstream_24"),
    ("Upstream_Mean", "Downstream_Mean", "upstream_to_downstream_mean"),
    ("DeltaP_13", "DeltaP_24", "differential_pair"),
]


def add_derived_pressure_channels(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add every pressure calculation supported by available source columns."""
    result = data.copy(); created: list[str] = []; skipped: list[str] = []
    calculations = {
        "Upstream_Mean": (("Pressure_1", "Pressure_2"), lambda d: (d.Pressure_1+d.Pressure_2)/2),
        "Downstream_Mean": (("Pressure_3", "Pressure_4"), lambda d: (d.Pressure_3+d.Pressure_4)/2),
        "DeltaP_13": (("Pressure_1", "Pressure_3"), lambda d: d.Pressure_1-d.Pressure_3),
        "DeltaP_24": (("Pressure_2", "Pressure_4"), lambda d: d.Pressure_2-d.Pressure_4),
        "Upstream_Pair_Difference": (("Pressure_1", "Pressure_2"), lambda d: d.Pressure_1-d.Pressure_2),
        "Downstream_Pair_Difference": (("Pressure_3", "Pressure_4"), lambda d: d.Pressure_3-d.Pressure_4),
    }
    for name, (required, function) in calculations.items():
        missing = [column for column in required if column not in result]
        if missing: skipped.append(f"{name}: missing {', '.join(missing)}")
        else: result[name] = function(result); created.append(name)
    if "Upstream_Mean" in result and "Downstream_Mean" in result:
        result["Turbine_DeltaP"] = result.Upstream_Mean-result.Downstream_Mean; created.append("Turbine_DeltaP")
    else: skipped.append("Turbine_DeltaP: requires Upstream_Mean and Downstream_Mean")
    return result, created, skipped


def add_dynamic_pressure_channels(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add run-median and cycle-median centered pressure channels."""
    result=data.copy(); created=[]
    required={"Run_ID","Upstream_Mean","Downstream_Mean"}
    if not required.issubset(result.columns): return result,created
    result["Upstream_Dynamic"]=np.nan; result["Downstream_Dynamic"]=np.nan
    valid=result.Run_ID.notna() & result.Run_ID.astype(str).ne("")
    for _,run in result[valid].groupby("Run_ID"):
        steady=run[run.Is_Steady_State] if "Is_Steady_State" in run else run
        for raw,dynamic in [("Upstream_Mean","Upstream_Dynamic"),("Downstream_Mean","Downstream_Dynamic")]:
            reference=float(steady[raw].median()) if steady[raw].notna().any() else np.nan
            result.loc[run.index,dynamic]=result.loc[run.index,raw]-reference
    result["Turbine_DeltaP_Dynamic"]=result.Upstream_Dynamic-result.Downstream_Dynamic
    created.extend(["Upstream_Dynamic","Downstream_Dynamic","Turbine_DeltaP_Dynamic"])
    if "Final_Cycle_Number" in result:
        result["Turbine_DeltaP_Cycle_Dynamic"]=np.nan
        cycle_rows=result[valid & result.Final_Cycle_Number.notna()]
        for _,cycle in cycle_rows.groupby(["Run_ID","Final_Cycle_Number"]):
            result.loc[cycle.index,"Turbine_DeltaP_Cycle_Dynamic"]=cycle.Turbine_DeltaP-cycle.Turbine_DeltaP.median()
        created.append("Turbine_DeltaP_Cycle_Dynamic")
    return result,created


def wrap_phase_degrees(phase: float) -> float:
    return float((phase + 180.0) % 360.0 - 180.0)


def lagged_cross_correlation(first: pd.Series, second: pd.Series, max_lag_samples: int) -> dict[str, Any]:
    """Positive lag means the second signal occurs after the first signal."""
    x = pd.to_numeric(first, errors="coerce").to_numpy(float); y = pd.to_numeric(second, errors="coerce").to_numpy(float)
    rows=[]
    for lag in range(-max_lag_samples, max_lag_samples+1):
        a,b = (x[:-lag],y[lag:]) if lag>0 else (x[-lag:],y[:lag]) if lag<0 else (x,y)
        valid=np.isfinite(a)&np.isfinite(b)
        corr=np.corrcoef(a[valid],b[valid])[0,1] if valid.sum()>=3 and np.std(a[valid])>0 and np.std(b[valid])>0 else np.nan
        rows.append((lag,corr))
    finite=[row for row in rows if np.isfinite(row[1])]
    if not finite: return {"lag_samples":np.nan,"maximum_correlation":np.nan,"lags":np.array([]),"correlations":np.array([])}
    best=max(finite,key=lambda row:abs(row[1]))
    return {"lag_samples":int(best[0]),"maximum_correlation":float(best[1]),"lags":np.array([r[0] for r in rows]),"correlations":np.array([r[1] for r in rows])}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator/denominator if np.isfinite(denominator) and denominator != 0 else np.nan


def classify_pair(row: dict[str, Any], config: dict[str, Any]) -> str:
    settings=config.get("stage5",{}); corr=float(row.get("Maximum_Correlation",np.nan)); amp=float(row.get("Peak_To_Peak_Amplitude_Ratio",np.nan)); offset=abs(float(row.get("Mean_Offset",np.nan))); phase=abs(float(row.get("Phase_Difference_Deg",np.nan)))
    if not np.isfinite(corr): return "insufficient evidence"
    if corr < float(settings.get("pair_minimum_correlation",0.5)): return "weak response"
    if phase > float(settings.get("pair_phase_tolerance_deg",30)): return "phase shifted"
    if offset > float(settings.get("pair_offset_tolerance",50)): return "consistent with offset"
    tolerance=float(settings.get("pair_amplitude_tolerance_fraction",0.35))
    if np.isfinite(amp) and abs(amp-1)>tolerance: return "consistent with gain difference"
    return "consistent"


def analyze_pressure_pairs(data: pd.DataFrame, run_periods: dict[str,float], sampling_interval_s: float, config: dict[str,Any]) -> tuple[pd.DataFrame,pd.DataFrame]:
    relationships=[]; consistency=[]; settings=config.get("stage5",{})
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        steady=run[run.Is_Steady_State] if "Is_Steady_State" in run else run
        period=run_periods.get(str(run_id),np.nan); lag_s=period*float(settings.get("maximum_lag_fraction",0.5)) if np.isfinite(period) else float(settings.get("fallback_maximum_lag_s",1.0)); max_lag=max(1,int(round(lag_s/sampling_interval_s)))
        for first,second,label in PRESSURE_PAIRS:
            versions=[("raw",first,second)]
            if f"{first}_Filtered" in steady and f"{second}_Filtered" in steady: versions.append(("filtered",f"{first}_Filtered",f"{second}_Filtered"))
            for version,first_column,second_column in versions:
                pair=steady[[first_column,second_column]].dropna(); x=pair[first_column]; y=pair[second_column]
                if pair.empty: continue
                pearson=float(pearsonr(x,y).statistic) if len(pair)>=3 and x.std()>0 and y.std()>0 else np.nan; spearman=float(spearmanr(x,y).statistic) if len(pair)>=3 else np.nan
                lag=lagged_cross_correlation(x,y,max_lag); lag_seconds=float(lag["lag_samples"]*sampling_interval_s) if np.isfinite(lag["lag_samples"]) else np.nan; phase=wrap_phase_degrees(360*lag_seconds/period) if np.isfinite(period) and np.isfinite(lag_seconds) else np.nan
                x_ptp=float(x.max()-x.min()); y_ptp=float(y.max()-y.min()); x_rms=float(np.sqrt(np.mean(x*x))); y_rms=float(np.sqrt(np.mean(y*y)))
                cycles=float((steady.TimeStamp.max()-steady.TimeStamp.min()).total_seconds()/period) if np.isfinite(period) and period>0 and len(steady)>1 else 0
                reasons=[]
                if cycles<float(settings.get("minimum_lag_analysis_cycles",2)): reasons.append(f"only {cycles:.2f} measured cycles")
                if not np.isfinite(lag["maximum_correlation"]) or abs(lag["maximum_correlation"])<float(settings.get("lag_reliable_correlation",.6)): reasons.append("weak maximum lagged correlation")
                correlations=np.asarray(lag["correlations"],float); lags=np.asarray(lag["lags"],int); best_lag=int(lag["lag_samples"]) if np.isfinite(lag["lag_samples"]) else 0
                separation=max(1,int(round(period*float(settings.get("conflicting_peak_separation_cycle_fraction",.10))/sampling_interval_s))) if np.isfinite(period) else 1
                distinct=np.abs(correlations[(np.abs(lags-best_lag)>=separation)&np.isfinite(correlations)])
                if len(distinct) and distinct.max()>=abs(lag["maximum_correlation"])*float(settings.get("conflicting_peak_fraction",.995)): reasons.append("multiple near-equal separated correlation peaks")
                reliability="reliable" if not reasons else "limited"
                row={"Run_ID":run_id,"Pair":label,"Signal_1":first,"Signal_2":second,"Data_State":"steady_state","Data_Version":version,"Zero_Lag_Pearson":pearson,"Zero_Lag_Spearman":spearman,"Maximum_Lagged_Correlation":lag["maximum_correlation"],"Maximum_Absolute_Lagged_Correlation":abs(lag["maximum_correlation"]) if np.isfinite(lag["maximum_correlation"]) else np.nan,"Signed_Lag_Samples":lag["lag_samples"],"Signed_Lag_Seconds":lag_seconds,"Phase_Degrees":360*lag_seconds/period if np.isfinite(period) and np.isfinite(lag_seconds) else np.nan,"Wrapped_Phase_Degrees":phase,"Lag_Sign_Convention":"Positive lag means Signal_2 occurs after Signal_1.","Measured_Cycle_s":period,"Lag_Search_Limit_s":max_lag*sampling_interval_s,"Reliability":reliability,"Reliability_Reason":"; ".join(reasons) if reasons else "sufficient cycles, correlation, and peak separation","Mean_Offset":float((y-x).mean()),"RMS_Difference":float(np.sqrt(np.mean((y-x)**2))),"Peak_To_Peak_Amplitude_Ratio":_safe_ratio(y_ptp,x_ptp),"RMS_Amplitude_Ratio":_safe_ratio(y_rms,x_rms),"Standard_Deviation_Ratio":_safe_ratio(float(y.std(ddof=0)),float(x.std(ddof=0)))}
                relationships.append(row)
                if label in {"upstream_pair","downstream_pair"} and version=="raw":
                    legacy={"Maximum_Correlation":row["Maximum_Lagged_Correlation"],"Phase_Difference_Deg":row["Wrapped_Phase_Degrees"],**row}
                    consistency.append({"Run_ID":run_id,"Pair":label,"Consistency_Classification":classify_pair(legacy,config),"Evidence":f"max_corr={row['Maximum_Lagged_Correlation']:.3f}; amplitude_ratio={row['Peak_To_Peak_Amplitude_Ratio']:.3f}; phase={row['Wrapped_Phase_Degrees']:.1f} deg; offset={row['Mean_Offset']:.3f}","Interpretation_Note":"Downstream sensors are evaluated against each other, not required to match upstream sensors."})
    return pd.DataFrame(relationships),pd.DataFrame(consistency)


def pressure_response_summary(data: pd.DataFrame, cycle_stats: pd.DataFrame, config: dict[str,Any]|None=None) -> pd.DataFrame:
    rows=[]
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        steady=run[run.Is_Steady_State]
        row={"Run_ID":run_id}
        if {"Upstream_Mean","Downstream_Mean"}.issubset(steady.columns):
            up=steady.Upstream_Mean.dropna(); down=steady.Downstream_Mean.dropna()
            row.update({"Raw_Upstream_PeakToPeak":float(np.ptp(up)),"Raw_Downstream_PeakToPeak":float(np.ptp(down)),"Robust_Upstream_P5_P95_Span":float(up.quantile(.95)-up.quantile(.05)),"Robust_Downstream_P5_P95_Span":float(down.quantile(.95)-down.quantile(.05))})
            row["Raw_Attenuation_Ratio"]=_safe_ratio(row["Raw_Downstream_PeakToPeak"],row["Raw_Upstream_PeakToPeak"]); row["Robust_Attenuation_Ratio"]=_safe_ratio(row["Robust_Downstream_P5_P95_Span"],row["Robust_Upstream_P5_P95_Span"])
        relevant=cycle_stats[(cycle_stats.Run_ID==run_id)&(cycle_stats.Signal.isin(["Upstream_Mean","Downstream_Mean","Turbine_DeltaP"]))] if not cycle_stats.empty else pd.DataFrame()
        for signal in relevant.Signal.unique() if not relevant.empty else []:
            amplitudes=relevant.loc[relevant.Signal==signal,"Peak_To_Peak"]
            row[f"{signal}_Median_Cycle_Peak_To_Peak"]=float(amplitudes.median()); row[f"{signal}_Cycle_Amplitude_MAD"]=float(np.median(np.abs(amplitudes-amplitudes.median())))
        up_cycles=relevant[relevant.Signal=="Upstream_Mean"][["Cycle_Number","Peak_To_Peak"]].rename(columns={"Peak_To_Peak":"up"}); down_cycles=relevant[relevant.Signal=="Downstream_Mean"][["Cycle_Number","Peak_To_Peak"]].rename(columns={"Peak_To_Peak":"down"})
        paired=up_cycles.merge(down_cycles,on="Cycle_Number"); ratios=paired["down"]/paired["up"] if not paired.empty else pd.Series(dtype=float)
        row["Median_Cycle_Upstream_PeakToPeak"]=float(up_cycles.up.median()) if not up_cycles.empty else np.nan; row["Median_Cycle_Downstream_PeakToPeak"]=float(down_cycles.down.median()) if not down_cycles.empty else np.nan; row["Median_Cycle_Attenuation_Ratio"]=float(ratios.median()) if not ratios.empty else np.nan; row["Cycle_Attenuation_MAD"]=float(np.median(np.abs(ratios-ratios.median()))) if not ratios.empty else np.nan
        tolerance=float((config or {}).get("stage5",{}).get("attenuation_disagreement_tolerance",.15)); row["Raw_Robust_Attenuation_Disagreement_Flag"]=bool(np.isfinite(row.get("Raw_Attenuation_Ratio",np.nan)) and np.isfinite(row.get("Robust_Attenuation_Ratio",np.nan)) and abs(row["Raw_Attenuation_Ratio"]-row["Robust_Attenuation_Ratio"])>tolerance)
        if "Turbine_DeltaP" in steady:
            raw=steady.Turbine_DeltaP.dropna(); dynamic=steady.Turbine_DeltaP_Dynamic.dropna() if "Turbine_DeltaP_Dynamic" in steady else pd.Series(dtype=float)
            row.update({"Raw_Turbine_DeltaP_Mean":float(raw.mean()),"Raw_Turbine_DeltaP_RMS":float(np.sqrt(np.mean(raw**2))),"Dynamic_Turbine_DeltaP_Mean":float(dynamic.mean()) if len(dynamic) else np.nan,"Dynamic_Turbine_DeltaP_RMS":float(np.sqrt(np.mean(dynamic**2))) if len(dynamic) else np.nan,"Dynamic_Turbine_DeltaP_PeakToPeak":float(np.ptp(dynamic)) if len(dynamic) else np.nan,"Dynamic_Turbine_DeltaP_P5_P95_Span":float(dynamic.quantile(.95)-dynamic.quantile(.05)) if len(dynamic) else np.nan})
            cycle_dynamic=cycle_stats[(cycle_stats.Run_ID==run_id)&(cycle_stats.Signal=="Turbine_DeltaP_Dynamic")]; row["Median_Cycle_Dynamic_Turbine_DeltaP_PeakToPeak"]=float(cycle_dynamic.Peak_To_Peak.median()) if not cycle_dynamic.empty else np.nan
            calibration=(config or {}).get("stage5",{}).get("pressure_calibration_status","unavailable_missing_calibration"); row["Absolute_DeltaP_Interpretation_Status"]="questionable_sensor_offset" if abs(float((steady.Pressure_4-steady.Pressure_3).mean()))>float((config or {}).get("stage5",{}).get("pair_offset_tolerance",50)) else calibration
            row["Absolute_DeltaP_Interpretation_Note"]="Raw mean is not an absolute physical turbine pressure drop without compatible calibration, units, zero reference, and sign convention."
        else: row["Absolute_DeltaP_Interpretation_Status"]="insufficient_channels"
        rows.append(row)
    return pd.DataFrame(rows)
