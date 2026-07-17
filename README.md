# HNEI OWC Analyzer

A reusable command-line project for intake and later engineering analysis of HNEI oscillating-water-column test-bench data.

This initial implementation covers **Stage 1–2 only**: project setup, CSV/Excel loading, configurable column aliases, timestamp parsing, and intake validation. Test segmentation, steady-state detection, signal processing, cycle/pressure/statistical analysis, plotting, and full reporting are intentionally deferred.

## Installation

From this repository in the VS Code terminal:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing virtual environment uses Python 3.12. On Windows, activate it with `.venv\\Scripts\\activate`.

## Run the Stage 1–2 intake audit

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

`--file-type` accepts `auto`, `csv`, `xlsx`, or `xls`. `--show-plots` and `--no-smoothing` are accepted for forward CLI compatibility but have no effect in Stage 1–2. If `--output` is omitted, a timestamped directory is created under `output/`. Existing non-empty output directories are protected from overwrite.

The intake run writes:

- `cleaned/full_annotated_data.csv` — original data plus row-order and quality-flag fields
- `tables/intake_audit.csv` — compact audit metrics
- `tables/data_quality_findings.csv` — immediate findings
- `run_log.txt` — detailed execution log

The source workbook is opened read-only and is never modified.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Stage 1–2 behavior

- Detects Excel worksheets and selects the requested sheet, or the most likely data sheet based on recognized headers and row count.
- Detects a likely units row without relying on fixed sample-workbook row boundaries.
- Maps aliases from `config/default_config.yaml`; ambiguous mappings are warned about and left unresolved.
- Parses text datetimes, native Excel datetimes, Unix timestamps, and Excel serial dates.
- Preserves source order in `Original_Row_Order`; sorts by timestamp only when reversals are present.
- Reports missing values, numeric conversion failures, duplicate rows/timestamps/record numbers, record gaps, timestamp reversals, timing gaps, jitter, and sampling statistics.
- Keeps questionable rows and adds `Quality_Flags` rather than deleting them.

## Project layout

The `src/` modules for Stage 3+ are placeholders so the planned project shape is visible without prematurely implementing later stages.

