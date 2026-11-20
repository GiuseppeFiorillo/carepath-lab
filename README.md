# CarePath Lab

An educational healthcare ML project studying recorded 30-day hospital
readmission in the historical UCI diabetes dataset. Built incrementally to learn
rigorous ML, SQL, software engineering, explainability and subgroup assessment.
**Not a clinical tool.**

## Current model and scope

The current model is **CatBoost with 41 predictors**, named `reviewed` in code.
It uses native categorical handling and training-fitted mode imputation for
`medical_specialty`. Payer and three training-constant medication fields are
excluded. See the [feature review](docs/FEATURE_REVIEW.md) for evidence and
tradeoffs; the revision does not claim improved performance or fairness.

Phases 0–2 are complete; Phase 3 is in progress. Subgroup metrics, fixed-bin
calibration, a patient-cluster recall comparison and numerical SHAP checks are
implemented, along with global/local SHAP plots and an
[XAI audit checkpoint](docs/MODEL_AUDIT_REPORT.md). Broader subgroup uncertainty
and the final integrated Model Audit Report remain pending.
The final test set has not been scored. Development validation has already been
inspected repeatedly; it is not external confirmation.

## Layout

```text
carepath_lab/   Shared data validation, features, modeling, audit and SHAP logic
scripts/        Current data preparation, training and audit commands
experiments/    Optional comparison, tuning and ablation studies
sql/            Full-source PostgreSQL schema
examples/       Earlier Python/SQL learning exercises
tests/          Synthetic unit and command tests
docs/           Dataset policies, decisions, evidence and reports
```

Run commands from the project root with `python -m ...`. No editable package
installation is required. Shared code lives in `carepath_lab`; commands do not
import utilities from other command modules.

## Local setup and source

Python 3.12+ (required by the pinned SHAP version); current checks ran on Apple
Silicon with Python 3.13.18. For a new
environment only:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On macOS, LightGBM requires Homebrew `libomp` (`brew install libomp` if absent).
Dependencies are pinned directly; a full transitive lock and clean-machine
reconstruction are not claimed. CatBoost supplies native SHAP computation;
the separate `shap` package supplies beeswarm and waterfall visualization.

Download the original [UCI Diabetes 130-US Hospitals, 1999–2008](https://doi.org/10.24432/C5230J)
files to `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv`.
Dataset attribution: Clore, Cios, DeShazo and Strack (2014), CC BY 4.0.

| Source file | Recorded SHA-256 |
| --- | --- |
| `diabetic_data.csv` | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |

The complete source has 101,766 encounters/71,518 patients. The analysis cohort
has 100,114 encounters after excluding discharge death codes 11/19/20/21.
Sources are preserved. `readmitted == "<30"` maps to 1; `">30"` and `"NO"` map to 0.
IDs are integrity/grouping keys, never predictors. See [cohort policy](docs/COHORT_POLICY.md).
Data, models, local artifacts, secrets and mentoring notes are excluded from Git.

## Current workflow

PostgreSQL is not required for this ML path. With source files available:

```bash
.venv/bin/python -m scripts.split_data
.venv/bin/python -m scripts.train_model --output-dir artifacts/current_model
.venv/bin/python -m scripts.audit_model --output-dir artifacts/current_audit
```

The split command creates or verifies saved patient-disjoint assignments.
If assignments and protocol match under a newer environment, it preserves the
original split creation metadata instead of overwriting it.
Train/validation/test contain 60,276/19,883/19,955 encounters, respectively.
All encounters of one patient stay in a single partition.

`train_model` fits only saved training and exports the fitted pipeline,
validation probabilities and metadata. It checks the reviewed schema/selection
record, reload prediction equality, SHAP shape/order/finiteness/additivity and
probability reconstruction on five examples, and input/code hashes.
It is a training command, not just a test or artifact checker.

`audit_model` reads `artifacts/current_model` by default, validates encounter,
patient and outcome alignment, then exports:

- `subgroup_metrics.csv`: support, confusion counts, recall/precision/FPR for
  recorded gender/age/race at thresholds .10/.20/.30/.50;
- `subgroup_calibration.csv`: fixed-bin predicted versus observed means;
- `bootstrap_recall_summary.csv`: Female-minus-Male recall at .20, with a
  1,000-replicate patient-cluster percentile interval, only if all draws are finite;
- `metadata.json`: settings, versions and input/code/output hashes.

Undefined metrics stay missing, not zero. These thresholds are exploratory,
not clinical operating points. Bootstrap conditions on the fitted model and
provides no simultaneous coverage of all comparisons.

Output directories must be new; no automatic replacement/deletion. For a repeat,
choose new destinations and pass the model directory explicitly to the audit:

```bash
.venv/bin/python -m scripts.train_model --output-dir artifacts/run_02_model
.venv/bin/python -m scripts.audit_model --source-artifacts artifacts/run_02_model --output-dir artifacts/run_02_audit
```

For Python use, load partitions/hash files from `carepath_lab.data`, construct
inputs with `build_model_inputs(..., feature_set="reviewed")`, and call
`carepath_lab.explainability.compute_native_shap` with the fitted pipeline.
The helper returns transformed inputs and native SHAP contributions plus their
reference, in raw log-odds. It does not fit, sample, export or plot.

To export explanations for the model in `artifacts/current_model`:

```bash
MPLBACKEND=Agg .venv/bin/python -m scripts.explain_model --output-dir artifacts/explanations
```

The command samples 1,000 validation encounters with seed42 and saves `sample.csv`
(IDs/order), `transformed_inputs.csv`, `shap_values.npz` (41 contributions,
separate reference values and feature names), `global_importance.png`,
`beeswarm.png`, `waterfall.png` and `metadata.json`. It verifies
the model fingerprint and source/split agreement with its training metadata,
then records file/code hashes, versions, sampling and interpretation limits.
NPZ arrays can be loaded with `allow_pickle=False`; CSV categorical labels must
be read without automatically treating literal `None` as a missing value.
All files remain in ignored artifacts; identifiers are not published.
Explanations are encounter-weighted and noncausal. The waterfall uses the first
sampled encounter; its position is recorded in metadata. The
[XAI report](docs/MODEL_AUDIT_REPORT.md) documents interpretation and descriptive
sensitivity across three samples.

## Optional data engineering and experiments

The [data audit](docs/DATA_AUDIT.md) and [data dictionary](docs/DATA_DICTIONARY.md)
document source checks. `scripts.profile_data` and
`scripts.audit_administrative_mappings` run without a database.

For PostgreSQL 18, create the local `carepath_lab` database and apply
`sql/001_create_encounters_raw.sql` and `sql/002_add_encounters_raw_checks.sql`
with `psql -v ON_ERROR_STOP=1 -f PATH`. Then run `scripts.load_raw_data` and
`scripts.verify_raw_import`. These use localhost:5432 and the local PostgreSQL
role; credentials/role provisioning are not automated. Loading requires an empty
table, preserves all 50 parsed fields, and commits one transaction. The live
verification is separate from the synthetic test suite.

Optional model studies are documented in [experiments/README.md](experiments/README.md).
They are not prerequisites for training the fixed current model. The historical
45-field `extended` and six-field `initial` schemas remain for those references;
[Phase 2 results](docs/PHASE_2_REPORT.md) are not relabeled as current-model results.
Earlier exercises remain under [examples/phase_1](examples/phase_1/README.md).
Superseded teaching commands are available in Git history rather than maintained
as additional entry points.

## Tests and interpretation limits

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Tests use synthetic data; they do not require source CSVs, a live database or
real-model retraining. Command integration is checked separately on local data.

Historical selection, incomplete outcome capture, missingness and unverified
field timestamps limit interpretation. `NO` means no recorded readmission,
not complete follow-up. SHAP describes model behavior, not causal effects.
Neither a small subgroup gap nor a confidence interval containing zero proves
fairness or equivalence. No diagnostic, treatment or deployment claim is made.

See the [roadmap](ROADMAP.md), [Model Audit Plan](docs/MODEL_AUDIT_PLAN.md),
[decisions](docs/DECISIONS.md) and [historical development report](docs/BASELINE_REPORT.md).
Code is licensed under [MIT](LICENSE); the dataset retains its separate license.
