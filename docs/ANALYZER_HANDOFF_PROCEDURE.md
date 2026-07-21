HNEI OWC Analyzer Handoff Procedure

1. Purpose
The hnei-owc-analyzer is a Python application developed to process data collected from the HNEI oscillating water column test bench.
The analyzer currently supports:
Excel input files
Actual timestamp-based timing
Numeric elapsed-time input
Elapsed-time reconstruction from RecNum
Operating-block detection
Encoder cycle detection
Steady-state selection
Pressure analysis
Torque analysis
Generator-voltage analysis
Signal-quality checks
Excel, CSV, Markdown, plot and reproducibility outputs
The analyzer should be treated as a research-analysis tool. It does not replace sensor calibration, physical inspection or engineering judgment.

2. Freeze the handoff version
Before preparing the handoff, open Terminal and move into the correct repository:
cd ~/Desktop/School/hnei-owc-analyzer

Confirm the location:
pwd
git rev-parse --show-toplevel

Both commands should return:
/Users/nicholasmorales/Desktop/School/hnei-owc-analyzer

2.1 Confirm the repository is current
git status
git pull

Run the full validation suite:
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q main.py src tests
git diff --check

Expected test result:
104 tests passed

If there are valid uncommitted analyzer changes, commit them before handoff:
git add config main.py src tests docs
git commit -m "Prepare analyzer for internship handoff"
git push

Do not run git add . from the home directory. The previous attempt nearly enlisted the entire Mac into one Git repository, which would have been historically significant for all the wrong reasons.

3. Record the repository information
Run:
git remote get-url origin
git rev-parse HEAD
git log -1 --format="%h %ad %s" --date=iso

Copy the results into this section.
Repository record
GitHub repository URL:
<PASTE OUTPUT OF git remote get-url origin>

Handoff commit hash:
<PASTE OUTPUT OF git rev-parse HEAD>

Commit date and message:
<PASTE OUTPUT OF git log -1 --format="%h %ad %s" --date=iso>

The previous analysis workbook recorded this commit:
b49e7d5556faa24428cc4dc78d76a64531d9bc39

Do not automatically use that as the final handoff commit. Use the current value returned by:
git rev-parse HEAD

because the reconstructed-time feature may have been committed afterward.

4. Record the required Python version
From the project directory, run:
.venv/bin/python --version

Record the exact result:
Validated Python version:
<PASTE VERSION HERE>

Also record the system Python:
python3 --version
which python3

The handoff should identify the exact Python version used for the successful 104-test validation. Do not merely write “Python 3,” because that covers a charmingly broad range of incompatible possibilities.
4.1 Record installed dependencies
Generate a complete environment snapshot:
.venv/bin/python -m pip freeze > requirements-handoff-lock.txt

Keep both:
requirements.txt
requirements-handoff-lock.txt

Their purposes differ:
requirements.txt lists the intended project dependencies.
requirements-handoff-lock.txt records the exact package versions used during handoff validation.
Add the lock file to Git only if project policy allows it:
git add requirements-handoff-lock.txt
git commit -m "Record validated handoff environment"
git push


5. New-computer setup procedure
The next person should follow these steps on a Mac.
5.1 Clone the repository
cd ~/Desktop
mkdir -p School
cd School
git clone <GITHUB_REPOSITORY_URL>
cd hnei-owc-analyzer

Replace <GITHUB_REPOSITORY_URL> with the URL recorded in Section 3.
5.2 Confirm the handoff commit
git rev-parse HEAD

Compare the result with the documented handoff commit hash.
To reproduce the exact handoff version:
git checkout <HANDOFF_COMMIT_HASH>

To return to the current main branch afterward:
git checkout main

5.3 Create the virtual environment
python3 -m venv .venv

Activate it:
source .venv/bin/activate

The terminal prompt should begin with something similar to:
(.venv)

Confirm that the virtual-environment interpreter is active:
which python
python --version

The Python path should end with:
hnei-owc-analyzer/.venv/bin/python

5.4 Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

5.5 Install dependencies
For normal development:
python -m pip install -r requirements.txt

For an exact reproduction of the handoff environment:
python -m pip install -r requirements-handoff-lock.txt

5.6 Verify installation
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests

Expected result:
104 tests passed


6. Project directory structure
The expected project structure is approximately:
hnei-owc-analyzer/
├── config/
│   ├── default_config.yaml
│   └── new_turbine_config.yaml
├── docs/
├── input/
├── output/
├── src/
├── tests/
├── main.py
├── requirements.txt
├── requirements-handoff-lock.txt
└── README.md

The following directories should normally remain outside Git tracking:
input/
output/
.venv/

Raw laboratory data and generated reports should not be committed to a public repository unless HNEI specifically authorizes it.

7. Example input files
7.1 Raw Campbell Scientific logger file
Example:
CR1000XSeries_Table1.dat

This is the original CR1000X TOA5 logger output.
It should be preserved unchanged as the raw source record.
7.2 Converted analyzer input
Example:
CR1000XSeries_Table1.xlsx

This workbook contains:
TimeStamp
RecNum
Pressure_1
Pressure_2
Pressure_3
Pressure_4
Torque
Encoder
Gen_V
Target_Cycle_s
Target_VFD_Hz
VFD_Command_mV

The validated example contained:
36,256 records
0.012-second median sampling interval
83.33 Hz nominal sampling rate

7.3 Recommended handoff example
Place an approved example file in an internal handoff folder:
HNEI_OWC_HANDOFF/
└── 07_TEST_DATA/
    └── example_analyzer_input/
        └── CR1000XSeries_Table1.xlsx

Do not place the raw dataset in the GitHub repository unless approved.
A reduced or anonymized example can be added to GitHub later if a portable demonstration dataset is needed.

8. CR1000X .dat conversion procedure
The raw Campbell Scientific file should remain untouched.
Copy it into the project’s input directory:
cp ~/Downloads/CR1000XSeries_Table1.dat \
   input/CR1000XSeries_Table1.dat

Convert it using Python:
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

source = Path("input/CR1000XSeries_Table1.dat")
destination = Path("input/CR1000XSeries_Table1.xlsx")

data = pd.read_csv(
    source,
    skiprows=[0, 2, 3],
    encoding="utf-8-sig",
    low_memory=False,
)

data = data.rename(
    columns={
        "TIMESTAMP": "TimeStamp",
        "RECORD": "RecNum",
    }
)

data.to_excel(destination, index=False)

print(f"Created: {destination}")
print(f"Rows: {len(data):,}")
print("Columns:")
for column in data.columns:
    print(f"  - {column}")
PY

Expected result for the validated example:
Created: input/CR1000XSeries_Table1.xlsx
Rows: 36,256

This procedure removes the TOA5 metadata and unit rows while preserving the measured data and real timestamps.

9. Example analyzer command
From the project root, run:
MPLCONFIGDIR=/tmp/hnei-mpl-cache \
.venv/bin/python main.py \
  --input input/CR1000XSeries_Table1.xlsx \
  --output output/2026-07-20_CR1000X_Table1_analysis \
  --config config/new_turbine_config.yaml \
  --file-type auto

The input contains real timestamps, so do not use:
--sample-interval-s

for this particular file.
The analyzer should prioritize the actual TimeStamp column.
9.1 Open the output
open output/2026-07-20_CR1000X_Table1_analysis

Open the summary workbook:
open output/2026-07-20_CR1000X_Table1_analysis/analysis_summary.xlsx


10. Time-source behavior
The analyzer selects a time source using this priority:
1. TimeStamp
2. Elapsed_Time_s
3. RecNum reconstruction

10.1 Actual timestamps
Use this whenever TimeStamp contains valid date-and-time values.
Benefits include:
Actual test date and time
Direct sampling-interval measurement
Detection of timestamp gaps
Sampling-jitter analysis
Separation of test sessions
10.2 Numeric elapsed time
If no calendar timestamp exists, the analyzer can use:
Elapsed_Time_s

This preserves real relative timing without inventing calendar dates.
10.3 Record-number reconstruction
If only RecNum exists, elapsed time may be reconstructed using:
[
t=(RecNum-RecNum_0)\Delta t
]
Example:
.venv/bin/python main.py \
  --input input/example_without_timestamps.xlsx \
  --output output/example_reconstructed_time \
  --config config/new_turbine_config.yaml \
  --sample-interval-s 0.012

This preserves gaps in record numbers.
It does not create artificial calendar timestamps.
When reconstructed timing is used, the report should disclose:
Time source: reconstructed_from_record_number
Assumed interval: 0.012 s


11. Configuration explanations
11.1 config/default_config.yaml
Use this for:
Legacy datasets
Default or historically established channel behavior
General analyzer defaults
Older test-bench configurations
This configuration should not automatically be assumed correct for the new turbine system.
11.2 config/new_turbine_config.yaml
Use this for:
Current new-turbine tests
Current column mappings
Current operating-block detection
Current steady-state criteria
Current timing fallback
New-turbine reporting
Example:
--config config/new_turbine_config.yaml

11.3 Command-line precedence
Where implemented, a command-line value should override the configuration value.
For example:
--sample-interval-s 0.012

overrides:
intake:
  fallback_sample_interval_s: 0.012

The command-line value is useful when the interval is known for a particular dataset and should be explicitly recorded in the analysis command.
11.4 Configuration changes
Do not change thresholds merely to force a test to pass.
Any threshold change should include:
Reason for change
Data used to justify it
Old value
New value
Effect on accepted cycles
Test results
Date and author
Keep experiment-specific configuration changes in a separate configuration profile rather than silently changing the established default.

12. Sample output package
The handoff should include one complete successful output directory.
Recommended package:
2026-07-20_CR1000X_Table1_analysis/

Preserve the entire directory, including:
analysis_summary.xlsx
reports/
tables/
plots/
logs/
metadata/
configuration snapshot
reproducibility information

Actual names may differ slightly depending on the analyzer version. Preserve the whole directory rather than choosing only the attractive graphs and abandoning the audit files to their bureaucratic fate.
12.1 Sample-output location
Store it internally as:
HNEI_OWC_HANDOFF/
└── 06_DATA_ANALYZER/
    └── sample_output/
        └── 2026-07-20_CR1000X_Table1_analysis/

12.2 What the sample output demonstrates
The sample package should demonstrate that the analyzer can:
Load a timestamped workbook
Identify test blocks
Detect encoder cycles
Select steady-state cycles
Produce pressure and torque summaries
Generate quality flags
Produce comparison tables
Save plots
Save reproducibility metadata

13. Known analyzer and dataset limitations
The handoff must include these limitations so the next person does not mistake produced numbers for validated physics.
13.1 Pressure calibration
Pressure channels are not yet fully confirmed in engineering units.
Current concerns include:
Pressure 3 and Pressure 4 disagreement
Pressure 4 offset
Unverified high/low port orientation
Unverified physical sensor locations
Unverified channel mapping
Unverified sensor ranges and scaling
Absolute differential-pressure conclusions should not be finalized until these are resolved.
13.2 Torque calibration
Torque trends are available, but the conversion to N·m remains unverified.
Current torque results should be described as relative channel values unless calibration is confirmed.
13.3 VFD scaling
The relationship between:
VFD_Command_mV
Target_VFD_Hz
actual VFD output frequency
encoder RPM
mechanical cycle period

has not been fully validated.
The analyzer currently records command values but cannot prove that the drive reached the requested output.
13.4 Low-speed encoder response
The 5.0 and 4.0-second test conditions produced nearly flat encoder data despite long hold durations.
Possible causes include:
Logger measurement mode
Encoder signal conditioning
Low-speed pulse handling
Channel configuration
Incorrect interpretation of the recorded encoder variable
Wiring or splitter issues
These runs should not be used for reliable encoder-derived cycle analysis until investigated.
13.5 Highest-speed response
The 2.5 and 2.0-second targets produced nearly the same measured cycle period.
This may indicate:
Incorrect VFD command scaling
Drive limiting
Motor or drivetrain limitation
Mechanical loading
Encoder-processing limitations
Actual system-speed plateau
Actual VFD output frequency and independent tachometer RPM are needed.
13.6 Generator voltage
Generator voltage is recorded, but:
Electrical loading is not fully documented
Current is not recorded
Generator power cannot be calculated from voltage alone
Different test sessions produced substantially different voltage levels
Do not report generator efficiency from the existing voltage column.
13.7 Multiple sessions in one file
The validated workbook contains separate test sessions divided by a long shutdown.
Aggregate statistics can be misleading if the shutdown gap is treated as part of continuous operation.
Future test sessions should preferably be stored in separate files or explicitly separated during analysis.
13.8 Reconstructed timing
For files without timestamps:
Absolute dates are unavailable
Sampling jitter cannot be measured
Frequency, phase, lag and drift-per-second results depend on the assumed interval
The assumed interval must be explicitly documented

14. Raw-data handling rules
Rule 1: Never overwrite the raw logger file
Preserve:
CR1000XSeries_Table1.dat

as the original source file.
Do not:
Delete header rows from it
Rename columns inside it
Open and resave it through Excel
Replace it with a converted file
Manually correct values
Rule 2: Work from copies
Use this chain:
Raw logger file
        ↓
Converted analysis workbook
        ↓
Generated analysis output

Example:
CR1000XSeries_Table1.dat
CR1000XSeries_Table1.xlsx
2026-07-20_CR1000X_Table1_analysis/

Rule 3: Preserve file provenance
For every dataset, record:
Test date
Test operator
Logger program
Logger serial number
Sampling interval
Test-bench configuration
Firmware version
Input filename
Input checksum
Analyzer commit hash
Configuration profile
Analysis command

Rule 4: Use descriptive filenames
Recommended raw-data convention:
YYYY-MM-DD_HNEI_OWC_<test-description>_RAW.dat

Example:
2026-07-20_HNEI_OWC_5to2s_extended_RAW.dat

Converted file:
2026-07-20_HNEI_OWC_5to2s_extended_INPUT.xlsx

Output directory:
2026-07-20_HNEI_OWC_5to2s_extended_ANALYSIS/

Rule 5: Keep test sessions separate
Do not append unrelated test days into a single logger file unless there is a specific reason.
Prefer:
2026-07-20_test.dat
2026-07-21_test.dat

rather than one file containing multiple days separated by long inactive periods.
Rule 6: Record stopped and operating periods
Each test should include:
Stopped baseline
Startup
Settling cycles
Steady operation
Shutdown
Ending baseline
Rule 7: Do not remove failed runs
Runs with flat encoders, unstable motion or sensor faults should remain in the raw dataset.
Flag them in the test log and analysis report. Failed tests are evidence, not clutter.
Rule 8: Preserve units
Do not rename a raw value as an engineering quantity unless calibration is known.
Examples:
Torque_raw
Pressure_4_raw
Gen_V_raw

are more honest than claiming:
Torque_Nm
Pressure_Pa
Generator_Voltage_V

without confirmed conversions.
Rule 9: Do not publish laboratory data without approval
The input and output folders should remain Git-ignored unless HNEI approves repository storage.
Use the internal handoff folder, secured Drive or approved institutional storage for real datasets.
Rule 10: Maintain checksums
Calculate a checksum for every preserved raw file:
shasum -a 256 input/CR1000XSeries_Table1.dat

Save the result in the test log:
SHA-256:
<PASTE HASH>

This proves whether the raw file changed after collection.

15. Reproducibility record for each analysis
Every important analysis should include a text file such as:
ANALYSIS_RUN_RECORD.txt

Use this template:
HNEI OWC ANALYSIS RUN RECORD

Test description:
Test date:
Test operator:

Raw input file:
Converted input file:
Raw input SHA-256:

GitHub repository:
Analyzer commit hash:
Python version:
Configuration file:

Time source:
Sampling interval:
Sample interval source:
Number of records:

Exact command:
MPLCONFIGDIR=/tmp/hnei-mpl-cache \
.venv/bin/python main.py \
  --input ...
  --output ...
  --config ...

Test result:
Output directory:

Known issues:
Notes:


16. Final handoff validation
Before declaring the analyzer handed off, complete this checklist.
Repository
Repository URL recorded
Latest changes committed
Latest changes pushed
Handoff commit hash recorded
Working tree clean
Repository clone tested
Environment
Exact Python version recorded
requirements.txt included
Exact dependency lock generated
Virtual-environment setup tested
All 104 tests pass
Compile check passes
Input
Raw .dat file preserved
Raw checksum recorded
Converted .xlsx created
Column names confirmed
Sample interval documented
Example input approved for handoff
Configuration
Default configuration explained
New-turbine configuration explained
Time-source priority explained
Threshold-change rules documented
Output
Sample command documented
Sample command rerun successfully
Sample output folder included
analysis_summary.xlsx opens successfully
Reports and plots included
Reproducibility metadata included
Limitations
Low-speed encoder issue documented
Pressure 3/4 issue documented
Torque calibration limitation documented
VFD scaling limitation documented
Generator-voltage limitation documented
Multi-session limitation documented
Reconstructed-time limitation documented

17. Final handoff statement
The HNEI OWC analyzer was validated at the documented handoff commit using the recorded Python environment and test suite. The program can reproducibly process approved CR1000X-derived Excel inputs and generate structured analysis packages. The software is operational, but several physical-system uncertainties remain, including low-speed encoder acquisition, pressure-channel calibration, torque calibration, VFD command scaling and generator electrical measurements. Future users should preserve raw data, use versioned configuration profiles and avoid modifying analysis thresholds solely to force acceptance of unstable operating conditions.

