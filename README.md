# HNEI OWC Analyzer

A reusable command-line project for intake and later engineering analysis of HNEI oscillating-water-column test-bench data.

This implementation covers **Stage 1–3**: project setup, reusable intake validation, test grouping, active-run detection, preliminary period estimation, cycle classification, and steady-state selection. Full signal processing, statistical/pressure analysis, final graph generation, and engineering reporting remain intentionally deferred.

## Installation

From this repository in the VS Code terminal:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing virtual environment uses Python 3.12. On Windows, activate it with `.venv\\Scripts\\activate`.

## Run the Stage 1–3 detection pipeline

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

The source workbook is opened read-only and is never modified.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Stage 1–3 behavior

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

## Project layout

Modules for Stage 4+ remain placeholders so later analysis is not implemented prematurely.
