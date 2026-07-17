# HNEI OWC Analyzer

A reusable command-line project for intake and later engineering analysis of HNEI oscillating-water-column test-bench data.

This implementation covers **Stage 1–4**: reusable intake validation, test grouping, active/steady-state detection, encoder processing and behavior classification, multi-method final cycle timing, and VFD-command verification. Pressure, torque, generator-voltage, correlation, final reporting, and Stage 5+ work remain intentionally deferred.

## Installation

From this repository in the VS Code terminal:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing virtual environment uses Python 3.12. On Windows, activate it with `.venv\\Scripts\\activate`.

## Run the Stage 1–4 analysis pipeline

```bash
python main.py --input input/hnei_owc_test_2026_07_14.xlsx
```

Optional arguments:

```bash
python main.py \
  --input input/data.xlsx \
  --output output/run_001 \
  --config config/default_config.yaml \
  --sheet Sheet1 \
  --file-type auto \
  --debug
```

`--file-type` accepts `auto`, `csv`, `xlsx`, or `xls`. Diagnostic plots are saved non-interactively. `--no-smoothing` is reserved for later analysis stages. If `--output` is omitted, a timestamped directory is created under `output/`. Existing non-empty output directories are protected from overwrite.

The intake run writes:

- `cleaned/full_annotated_data.csv` — original data plus row-order and quality-flag fields
- `tables/intake_audit.csv` — compact audit metrics
- `tables/data_quality_findings.csv` — immediate findings
- `run_log.txt` — detailed execution log
- `tables/run_boundaries.csv` — recorded or inferred operating blocks
- `tables/preliminary_period_estimates.csv` — peak and autocorrelation estimates
- `tables/steady_state_selection.csv` and `cycle_classification.csv`
- `cleaned/steady_state_data.csv` and per-run all/steady CSV files
- `graphs/stage3_diagnostics/` — detection-verification plots only
- `tables/encoder_behavior_classification.csv`
- `tables/encoder_cycle_intervals.csv`, `encoder_cycle_method_comparison.csv`, and `encoder_cycle_summary.csv`
- `tables/vfd_command_verification.csv`
- `graphs/stage4_diagnostics/` — encoder timing and reconstructed-command verification plots

The source workbook is opened read-only and is never modified.

## Configuration profiles

- `config/legacy_config.yaml` preserves the verified legacy 5–8 second inference targets and VFD command equation.
- `config/new_turbine_config.yaml` supports expected 2.0, 2.5, and 3.0 second inference targets. Its VFD verification is intentionally disabled because no verified voltage-to-frequency constants have been supplied.
- Recorded positive floating-point `Target_Cycle_s` values are always preferred and are not restricted to either expected-target list. Expected targets are used only for inference.

The period searches cover at least 1.5–10 seconds. Peak spacing and smoothing use fractions of the relevant cycle period, with configured minimum and maximum windows, so decimal and shorter periods are not forced through legacy 5–8 second windows.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Stage 1–4 behavior

- Detects Excel worksheets and selects the requested sheet, or the most likely data sheet based on recognized headers and row count.
- Detects a likely units row without relying on fixed sample-workbook row boundaries.
- Maps aliases from `config/default_config.yaml`; ambiguous mappings are warned about and left unresolved.
- Parses text datetimes, native Excel datetimes, Unix timestamps, and Excel serial dates.
- Preserves source order in `Original_Row_Order`; sorts by timestamp only when reversals are present.
- Reports missing values, numeric conversion failures, duplicate rows/timestamps/record numbers, record gaps, timestamp reversals, timing gaps, jitter, and sampling statistics.
- Keeps questionable rows and adds `Quality_Flags` rather than deleting them.
- Groups contiguous recorded `Target_Cycle_s` values when available, including repeated targets as separate runs.
- Otherwise detects active blocks using configurable rolling encoder and supporting-signal activity.
- Estimates preliminary periods using both peak timing and autocorrelation, then labels expected periods only above the configured confidence threshold.
- Adds `Run_ID`, inferred-target/source fields, `Operating_State`, cycle number, steady-state status, and confidence while retaining every original row.
- Supports manual run-boundary and steady-state overrides in the YAML configuration.
- Reports both median-derived and effective-mean sampling rates, plus sampling-interval CV as a ratio and percent.
- Preserves raw encoder values while adding median-filtered, smoothed, detrended, and spike-flag fields for timing diagnostics.
- Calculates peak, trough, autocorrelation, FFT, and zero-crossing period estimates and selects only sufficiently agreeing methods.
- Reports all cycle intervals and flags timing outliers without deleting them.
- Reconstructs uncapped/capped VFD commands from recorded or inferred targets and clearly labels reconstructed values.
- Separates physical cycle frequency (`1 / cycle time`) from VFD-equivalent frequency (`120 / cycle time`).
- Reports cycle-time variability separately for all intervals and outlier-excluded valid intervals. Population standard deviation/CV describe only the observed intervals (`ddof=0`); sample standard deviation/CV estimate variability beyond those observations (`ddof=1`). Each CV uses its matching standard deviation. Sample variability is primary when at least two intervals exist; otherwise sample standard deviation and CV are unavailable. Percent fields equal ratio × 100.
- `Timing_Method_Confidence` describes confidence in the selected timing estimate from method agreement, signal quality, and available events. It is not a cycle-stability metric; stability is represented by standard deviation, CV ratio/percent, and peak-to-peak timing range.
- `Final_Selected_Period_s` is the method-selected timing result used for VFD error calculations. It may differ from the arithmetic `Valid_Interval_Mean_s`; `Selection_Reason` explains why. Cycle variability always comes from the cycle-level intervals, never from the selected multi-method period.

CSV files retain calculation precision. Terminal and engineering-summary displays use approximately three decimals for periods, frequencies, percentages, and CV percent, three or four decimals for errors, and two or three decimals for command voltage. With a typical 0.012-second sample interval, averaging cycles can improve a mean estimate but does not create microsecond measurement resolution.

For the legacy 5-second case, the nominal-to-capped expectation difference is about 0.020921 seconds. Numerical closeness alone is not treated as evidence: sampling interval, timing variation, interval count, and recorded command/frequency availability are considered before claiming the two expectations are distinguishable.

## Project layout

Modules for Stage 4+ remain placeholders so later analysis is not implemented prematurely.
