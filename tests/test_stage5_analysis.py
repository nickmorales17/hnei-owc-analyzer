import unittest

import numpy as np
import pandas as pd

from src.correlation_analysis import phase_relationship
from src.file_loader import load_config
from src.pressure_analysis import add_derived_pressure_channels, add_dynamic_pressure_channels, analyze_pressure_pairs, lagged_cross_correlation, pressure_response_summary, wrap_phase_degrees
from src.quality_checks import apply_quality_checks, robust_spike_flags
from src.statistics_analysis import cycle_level_statistics, descriptive_statistics, generator_summary, numeric_statistics, regression_metrics, torque_summary


def synthetic(period=2.5, n=1000, dt=.01):
    t=np.arange(n)*dt; phase=2*np.pi*t/period
    return pd.DataFrame({"TimeStamp":pd.Timestamp("2026-01-01")+pd.to_timedelta(t,unit="s"),"Run_ID":"run_001","Operating_State":"steady_state","Is_Steady_State":True,"Final_Cycle_Number":np.floor(t/period).astype(int)+1,"Pressure_1":100+20*np.sin(phase),"Pressure_2":102+18*np.sin(phase-.1),"Pressure_3":70+10*np.sin(phase-.3),"Pressure_4":71+9*np.sin(phase-.35),"Torque":5+2*np.sin(phase-.2),"Encoder":50*np.sin(phase),"Gen_V":3+.02*t})


class Stage5AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config("config/default_config.yaml")

    def test_derived_pressure_channels_are_exact_and_raw_unchanged(self):
        source=synthetic(); raw=source.Pressure_1.copy(); result,created,skipped=add_derived_pressure_channels(source)
        self.assertTrue(np.allclose(result.Turbine_DeltaP,(source.Pressure_1+source.Pressure_2-source.Pressure_3-source.Pressure_4)/2)); self.assertIn("DeltaP_13",created); self.assertFalse(skipped); pd.testing.assert_series_equal(source.Pressure_1,raw)

    def test_missing_pressure_columns_are_skipped(self):
        result,created,skipped=add_derived_pressure_channels(pd.DataFrame({"Pressure_1":[1.]})); self.assertFalse(created); self.assertTrue(any("missing" in x for x in skipped)); self.assertNotIn("Turbine_DeltaP",result)

    def test_lag_sign_and_phase_wrapping(self):
        x=pd.Series(np.sin(np.arange(400)*.1)); y=x.shift(5)
        result=lagged_cross_correlation(x,y,20); self.assertEqual(result["lag_samples"],5); self.assertEqual(wrap_phase_degrees(190),-170)

    def test_pressure_pair_metrics_and_decimal_period(self):
        data,_,_=add_derived_pressure_channels(synthetic(2.5)); relationships,consistency=analyze_pressure_pairs(data,{"run_001":2.5},.01,self.config)
        self.assertEqual(len(relationships),6); self.assertEqual(len(consistency),2); self.assertTrue((relationships.Measured_Cycle_s==2.5).all()); self.assertIn("Positive lag",relationships.Lag_Sign_Convention.iloc[0])

    def test_phase_relationship_detects_delayed_second_signal(self):
        t=np.arange(1000)*.01; x=pd.Series(np.sin(2*np.pi*t/2)); y=x.shift(25)
        result=phase_relationship(x,y,2,.01); self.assertAlmostEqual(result["Lag_s"],.25,places=2); self.assertAlmostEqual(result["Phase_Deg"],45,places=0)

    def test_numeric_and_descriptive_statistics(self):
        stats=numeric_statistics(pd.Series([1,2,3]),2); self.assertEqual(stats["Peak_To_Peak"],2); self.assertAlmostEqual(stats["RMS"],np.sqrt(14/3))
        data,_,_=add_derived_pressure_channels(synthetic()); table=descriptive_statistics(data); self.assertIn("steady_state",table.Section.unique()); self.assertIn("Turbine_DeltaP",table.Signal.unique())

    def test_cycle_torque_metrics_use_cycle_values(self):
        data,_,_=add_derived_pressure_channels(synthetic()); cycles=cycle_level_statistics(data,self.config); torque=torque_summary(data,cycles,{"run_001":2.5})
        self.assertGreater(len(cycles[cycles.Signal=="Torque"]),1); self.assertAlmostEqual(torque.Measured_Cycle_Frequency_Hz.iloc[0],.4); self.assertGreater(torque.RMS_Torque.iloc[0],0)

    def test_regression_labels_data_level_and_four_run_caution(self):
        result=regression_metrics(pd.Series([1,2,3,4]),pd.Series([2,4,6,8]),"run summary","test")
        self.assertEqual(result["Data_Level"],"run summary"); self.assertAlmostEqual(result["R_Squared"],1); self.assertIn("exploratory",result["Caution"])

    def test_spike_detection_retains_raw_and_adds_filtered(self):
        x=pd.Series(np.sin(np.arange(501)*.1)); x.iloc[250]=100
        flags,severity,filtered=robust_spike_flags(x,.01,self.config); self.assertTrue(flags.iloc[250]); self.assertLess(filtered.iloc[250],10); self.assertEqual(x.iloc[250],100); self.assertGreater(severity.iloc[250],8)

    def test_quality_finds_offset_constant_and_drift(self):
        data=synthetic(); data["Pressure_4"]=data.Pressure_3+100; data["Gen_V"]=3.; data.loc[500,"Torque"]=100
        annotated,findings,counts=apply_quality_checks(data,.01,self.config); checks=set(findings.Check); self.assertIn("pressure_pair_offset",checks); self.assertIn("near_constant_signal",checks); self.assertIn("isolated_spikes",checks); self.assertIn("Torque_Filtered",annotated); self.assertFalse(counts.empty)

    def test_generator_drift_is_transparently_classified(self):
        data=synthetic(); summary=generator_summary(data,{"run_001":2.5},self.config); self.assertGreater(summary.Linear_Drift_Per_s.iloc[0],0); self.assertIn(summary.Combined_GenV_Classification.iloc[0],["drifting_nonperiodic","periodic_with_drift"]); self.assertIn("undocumented",summary.Physical_Channel_Warning.iloc[0])

    def test_profiles_inherit_stage5_without_vfd_invention(self):
        legacy=load_config("config/legacy_config.yaml"); new=load_config("config/new_turbine_config.yaml")
        self.assertIn("stage5",legacy); self.assertIn("stage5",new); self.assertTrue(legacy["vfd_command"]["enabled"]); self.assertFalse(new["vfd_command"]["enabled"])

    def test_raw_robust_and_cycle_attenuation_separate_spike_effect(self):
        data=synthetic(); data.loc[500,"Pressure_4"]+=1000; data,_,_=add_derived_pressure_channels(data); data,_=add_dynamic_pressure_channels(data); cycles=cycle_level_statistics(data,self.config); result=pressure_response_summary(data,cycles,self.config).iloc[0]
        self.assertGreater(result.Raw_Attenuation_Ratio,result.Robust_Attenuation_Ratio); self.assertTrue(result.Raw_Robust_Attenuation_Disagreement_Flag); self.assertTrue(np.isfinite(result.Median_Cycle_Attenuation_Ratio)); self.assertTrue(np.isfinite(result.Cycle_Attenuation_MAD))

    def test_dynamic_delta_p_is_offset_invariant_and_uncalibrated(self):
        base=synthetic(); shifted=base.copy(); shifted["Pressure_4"]+=500
        base,_,_=add_derived_pressure_channels(base); shifted,_,_=add_derived_pressure_channels(shifted); base,_=add_dynamic_pressure_channels(base); shifted,_=add_dynamic_pressure_channels(shifted)
        np.testing.assert_allclose(base.Turbine_DeltaP_Dynamic,shifted.Turbine_DeltaP_Dynamic); cycles=cycle_level_statistics(shifted,self.config); result=pressure_response_summary(shifted,cycles,self.config).iloc[0]; self.assertEqual(result.Absolute_DeltaP_Interpretation_Status,"questionable_sensor_offset"); self.assertAlmostEqual(result.Dynamic_Turbine_DeltaP_Mean,0,places=10)

    def test_generator_periodic_with_drift(self):
        data=synthetic(period=2.5,n=3000); t=np.arange(len(data))*.01; data["Gen_V"]=3+.1*t+np.sin(2*np.pi*t/2.5)
        result=generator_summary(data,{"run_001":2.5},self.config).iloc[0]; self.assertEqual(result.Drift_Classification,"significant_drift"); self.assertEqual(result.Periodicity_Classification,"periodic"); self.assertEqual(result.Combined_GenV_Classification,"periodic_with_drift")

    def test_generator_periodic_without_drift(self):
        data=synthetic(period=2.5,n=3000); t=np.arange(len(data))*.01; data["Gen_V"]=3+np.sin(2*np.pi*t/2.5)
        result=generator_summary(data,{"run_001":2.5},self.config).iloc[0]; self.assertEqual(result.Combined_GenV_Classification,"periodic_without_significant_drift"); self.assertTrue(np.isfinite(result.Dominant_Period_s))

    def test_generator_nonperiodic_drift(self):
        data=synthetic(n=3000); t=np.arange(len(data))*.01; data["Gen_V"]=3+.1*t
        result=generator_summary(data,{"run_001":2.5},self.config).iloc[0]; self.assertEqual(result.Combined_GenV_Classification,"drifting_nonperiodic")

    def test_lag_schema_formula_and_weak_reliability(self):
        data=synthetic(n=300); rng=np.random.default_rng(4); data["Pressure_2"]=rng.normal(size=len(data)); data,_,_=add_derived_pressure_channels(data)
        result,_=analyze_pressure_pairs(data,{"run_001":2.5},.01,self.config); row=result[(result.Pair=="upstream_pair")&(result.Data_Version=="raw")].iloc[0]
        self.assertEqual(row.Lag_Sign_Convention,"Positive lag means Signal_2 occurs after Signal_1."); self.assertAlmostEqual(row.Phase_Degrees,360*row.Signed_Lag_Seconds/row.Measured_Cycle_s); self.assertGreaterEqual(row.Wrapped_Phase_Degrees,-180); self.assertLess(row.Wrapped_Phase_Degrees,180); self.assertEqual(row.Reliability,"limited")

    def test_dynamic_annotation_preserves_rows_and_raw_values(self):
        source=synthetic(); raw=source[["Pressure_1","Pressure_2","Pressure_3","Pressure_4"]].copy(); result,_,_=add_derived_pressure_channels(source); result,_=add_dynamic_pressure_channels(result)
        self.assertEqual(len(result),len(source)); pd.testing.assert_frame_equal(result[raw.columns],raw)


if __name__ == "__main__":
    unittest.main()
