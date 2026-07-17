"""Reusable lag/phase helpers for torque and other Stage 5 relationships."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .pressure_analysis import lagged_cross_correlation,wrap_phase_degrees


def phase_relationship(first:pd.Series,second:pd.Series,period_s:float,sampling_interval_s:float,max_lag_fraction:float=0.5) -> dict[str,Any]:
    if not np.isfinite(period_s) or period_s<=0: return {"Lag_s":np.nan,"Phase_Deg":np.nan,"Maximum_Correlation":np.nan,"Phase_Confidence":"unavailable_cycle_timing"}
    result=lagged_cross_correlation(first,second,max(1,int(round(period_s*max_lag_fraction/sampling_interval_s)))); lag_s=result["lag_samples"]*sampling_interval_s if np.isfinite(result["lag_samples"]) else np.nan; confidence="high" if np.isfinite(result["maximum_correlation"]) and abs(result["maximum_correlation"])>=.6 else "low"
    return {"Lag_s":lag_s,"Phase_Deg":wrap_phase_degrees(360*lag_s/period_s) if np.isfinite(lag_s) else np.nan,"Maximum_Correlation":result["maximum_correlation"],"Phase_Confidence":confidence,"Lag_Sign_Convention":"positive lag means second signal occurs after first"}


def torque_phase_summary(data:pd.DataFrame,run_periods:dict[str,float],sampling_interval_s:float,config:dict[str,Any]) -> pd.DataFrame:
    rows=[]; fraction=float(config.get("stage5",{}).get("maximum_lag_fraction",.5))
    for run_id,run in data[data.Run_ID.notna() & data.Run_ID.astype(str).ne("")].groupby("Run_ID"):
        steady=run[run.Is_Steady_State]; period=run_periods.get(str(run_id),np.nan)
        for reference in ["Encoder","Turbine_DeltaP"]:
            if reference in steady and "Torque" in steady: rows.append({"Run_ID":run_id,"Relationship":f"Torque relative to {reference}",**phase_relationship(steady[reference],steady.Torque,period,sampling_interval_s,fraction)})
    return pd.DataFrame(rows)
