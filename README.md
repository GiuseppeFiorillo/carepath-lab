# CarePath Lab

An educational healthcare data science project studying recorded hospital
readmission in the historical UCI diabetes dataset. Built incrementally to learn
rigorous ML, SQL and software engineering. It is not a clinical tool.

## Status and scope

Phases 0–2 are complete. The [Phase 2 report](docs/PHASE_2_REPORT.md) is the current
modeling reference. The [development history](docs/BASELINE_REPORT.md) preserves the six-feature
logistic/dummy reference and records the subsequent 45-feature logistic,
Random Forest and LightGBM grouped-CV and saved-validation comparisons, plus
CatBoost native-category and controlled imputation experiments. Tree-model
numerical scaling has been removed; earlier scaled results remain labeled as
historical snapshots. The bounded model/imputation search is finished for Phase 2:
CatBoost with training-fitted medical_specialty mode imputation and no missingness
indicator is the AP-selected candidate, not a proven superior or clinical model.
The final test set has not been scored. The selected training-fitted CatBoost
pipeline, probabilities and execution metadata are saved locally under ignored
artifacts/. Aggregate validation metrics, paired bootstrap intervals and PR/
calibration figures are exported in the report. The validation has already been
inspected and is not an untouched confirmatory sample. Next: Phase 3 model
explanation and experiment tracking; the final test remains reserved.
See the [phase checklist](ROADMAP.md) for completion criteria.

| Dataset / analysis population | Encounters | Patients / outcomes |
| --- | ---: | --- |
| Original source and raw table | 101,766 | 71,518 patients; 50 columns |
| Initial analysis cohort | 100,114 | 11,357 positives; 88,757 negatives; 11.34% positives |

The recorded target is `<30` versus `>30`/`NO`. The initial cohort excludes
`discharge_disposition_id` codes `11`, `19`, `20`, `21`, while preserving repeated
patients and the complete original data. Identifiers are metadata/grouping keys,
never predictive features. See [cohort policy](docs/COHORT_POLICY.md) for exact
rules and limitations. These counts are descriptive, not model metrics.

## Repository layout

```text
carepath_lab/       Shared validation, data preparation, modeling and resampling
scripts/           Data, split and baseline commands
sql/               Two full-source schema files
tests/            Synthetic unit/orchestration tests
docs/             Audit, dictionary, decisions and current progress
examples/phase_1/   Optional earlier Python/SQL learning exercises
```

`carepath_lab` is a small importable package; `scripts` contains commands that
use it. Run commands with `python -m ...` from the project root. No editable
package installation or additional framework is required at this stage.

## Local setup

Tested on macOS 26.5.2 arm64 with Python 3.13.5, pandas 3.1.0 and Psycopg 3.3.6.
Scikit-learn 1.9.1 is installed for Phase 2; its dependency checks and a synthetic
grouped-split smoke check passed without changing existing NumPy/pandas/Psycopg
versions. The project targets Python 3.11+. PostgreSQL 18.6 was installed separately through
Homebrew. The current environment is tested; a fresh-machine reconstruction has
not been verified. Direct dependency pins are not a complete transitive lock.

LightGBM 4.7.0 is pinned for the Phase 2 boosting comparison. On macOS it also
requires the external OpenMP runtime (`libomp`); if absent, install it through
Homebrew with `brew install libomp` before verifying the Python import.
The current machine already had libomp, so no system package was changed.
The ARM64 wheel, sparse synthetic fit/prediction and `pip check` passed locally.

CatBoost 1.2.10 is also pinned. Its native-category procedure receives named
categorical columns without external one-hot encoding or numerical scaling;
optional imputation is fitted separately inside each patient-grouped fold.

For a new Python environment only:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Skip creation/installation if your environment is already prepared. Examples
below use `.venv/bin/python` explicitly, so activation is optional. Git,
PostgreSQL and `psql` are external prerequisites, not Python packages.

## Source data

Download the original files from [UCI Diabetes 130-US Hospitals for Years
1999–2008](https://doi.org/10.24432/C5230J) into:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

Dataset: Clore, Cios, DeShazo and Strack (2014), CC BY 4.0. Raw/processed data,
secrets, virtual environments and trained artifacts are excluded by `.gitignore`.
The source files remain unchanged; downloads are not automated.

| File | Recorded SHA-256 |
| --- | --- |
| `diabetic_data.csv` | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |

These identify the local files; they do not independently establish authenticity
or clinical accuracy. Current contents have not been compared against a fresh
official download.

## Full-source PostgreSQL import

On the tested Apple Silicon Homebrew setup, if needed:

```bash
export PATH="/opt/homebrew/opt/postgresql@18/bin:$PATH"
brew services run postgresql@18
```

The scripts use `localhost:5432`, database `carepath_lab`, and the default local
PostgreSQL role. Role/password provisioning is not automated. For a new database
and new raw table, execute once:

```bash
psql -h localhost -p 5432 -d postgres -c "CREATE DATABASE carepath_lab;"
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f sql/001_create_encounters_raw.sql
psql -h localhost -p 5432 -d carepath_lab -v ON_ERROR_STOP=1 -f sql/002_add_encounters_raw_checks.sql
.venv/bin/python -m scripts.load_raw_data
```

**Existing populated database:** skip creation, schema files and loading. The
renamed schema files are the same definitions previously named `009`/`010`, not
new migrations to apply again. No existing table was dropped or changed during
repository cleanup.

The loader validates column names/order and records, preserves all 50 parsed
source fields as `TEXT`, and uses parameterized inserts in one transaction. The
primary key, non-null fields and minimal CHECK constraints guard identifiers
and target categories. A propagated error aborts the transaction; success is
printed after commit. A populated destination is rejected, without automatic
replacement or deletion. Run one loader at a time: the emptiness check is not a
concurrency lock. Analysis cohort filtering is separate from raw ingestion.

## Data audit and import commands

| Command module | Purpose | PostgreSQL required? |
| --- | --- | --- |
| `scripts.load_raw_data` | Load a validated source into an empty raw table | Yes; writes only during an intentional import |
| `scripts.verify_raw_import` | Compare source/header and every stored field by encounter ID | Yes; reads only |
| `scripts.profile_data` | Audit markers, numeric values, categories, duplicates and cohort counts | No |
| `scripts.audit_administrative_mappings` | Audit mapping coverage and unavailable-information codes | No |

With source CSVs available, run audits and verification:

```bash
.venv/bin/python -m scripts.profile_data
.venv/bin/python -m scripts.audit_administrative_mappings
.venv/bin/python -m scripts.verify_raw_import
```

The profiler regenerates the aggregate
`docs/reports/missing_marker_profile.csv`; original sources are preserved. Exact
import verification must report 101,766 database records and zero missing,
unexpected or mismatched records. This establishes import fidelity, not source
quality, complete follow-up or clinical usefulness.

## Tests and learning examples

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests use synthetic data and mocked database connections; original source
CSVs and a running database are not required for that suite. Live PostgreSQL
verification is a separate acceptance check. Neither the unit suite nor the live
comparison is an exhaustive clinical validation or a full automated loader
rollback integration suite.

Earlier CSV loops, five-column staging, JOIN/CTE/HAVING queries and the temporary
table rollback demonstration are preserved in
[examples/phase_1](examples/phase_1/README.md). They are optional learning artifacts,
not prerequisites for the current pipeline. The duplicate assistant-written
inspector and redundant empty-staging checker were removed; useful header checks
were consolidated in the shared validator and tests.

## Frozen Phase 2 model comparison

With the original source and verified saved split artifacts from
`scripts.split_data` available, run the commands below. On a new checkout with
the downloaded CSV, first create the saved splits; matching existing artifacts
are reused, incompatible ones are refused. PostgreSQL is not required for this
modeling path.

```bash
.venv/bin/python -m scripts.split_data
.venv/bin/python -m scripts.evaluate_final_models \
  --output-dir artifacts/phase2_validation_repeat --bootstrap 1000
```

Choose a new output directory: existing destinations are refused. This fits the
four frozen pipelines and prior dummy on saved training, then exports development
validation metrics/calibration, 1,000 paired patient-bootstrap replicates per
metric, figures, prediction artifacts and the selected fitted pipeline. It does
not score the final test or select a new winner from validation. Raw source
predictor count is 45; the selected pipeline imputes only medical_specialty.
Matplotlib/joblib are pinned for report figures and local model persistence.

`scripts.train_baseline` and `scripts.evaluate_models` preserve the initial
teaching comparisons. `scripts.tune_lightgbm` reproduces the eight-setting search;
its output destination is currently set in main and must be new. These commands
are documented history, not prerequisites to rerun the frozen comparison.

## Scientific limits and next step

Historical 1999–2008 observations do not establish present-day clinical utility
or causal effects. `NO` means no recorded readmission, not verified complete
follow-up. Field-level availability at discharge is assumed or uncertain; the
current 45-feature comparison uses the declared assumptions/exclusions. Missing markers, nonmeasurement and
unavailable codes have field-specific meanings and are not automatically
interchangeable. No diagnoses, treatment advice or patient triage are provided.

The closing selected CatBoost validation AP is 0.205999 (conditional patient-
bootstrap 95% interval 0.188982–0.227624), at prevalence 0.112307. AP differences
versus RF/LightGBM remain unresolved by paired intervals. These are development
results from the same source, not independent external or prospective validation.

## License and publication

Code is under the [MIT license](LICENSE), selected by the project author.
The public dataset retains CC BY 4.0 and its attribution above. Data, fitted
models, secrets and local personal notes are ignored. The repository has been
prepared for review; no commit, push or publication was performed by the assistant.
The final test remains reserved until model/analysis choices are fixed.

## Project documentation

- [Data audit](docs/DATA_AUDIT.md): observed findings and verification evidence.
- [Data dictionary](docs/DATA_DICTIONARY.md): meanings, availability assumptions and literature notes.
- [Cohort policy](docs/COHORT_POLICY.md): initial cohort, target and preprocessing contract.
- [Phase 2 report](docs/PHASE_2_REPORT.md): current comparison, uncertainty and limitations.
- [Development history](docs/BASELINE_REPORT.md): earlier baseline experiments.
- [Decisions](docs/DECISIONS.md): scientific and engineering choices.
- [Project brief](PROJECT_BRIEF.md) and [roadmap](ROADMAP.md): scope and learning phases.

Personal mentoring prompts, session histories and local progress notes are not
part of the published repository. `AGENTS.md` documents collaboration and
scientific guardrails; it is intentionally versionable. The code license is MIT
(see LICENSE); no public repository, commit, remote or release was created here.
