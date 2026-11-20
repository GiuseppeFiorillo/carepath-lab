# Data dictionary

## `patient_nbr`

`patient_nbr` is a patient identifier. Multiple encounters may share the same 
identifier.

Retain it to group encounters when constructing training, validation and test
splits. If the goal is to evaluate generalization to unseen patients, allowing
the same patient in multiple splits can produce an optimistic estimate because
their encounters may share persistent characteristics.

Exclude `patient_nbr` from predictive features. Keep it separately for grouping
during cross-validation, checking split disjointness and, when appropriate,
patient-level aggregation or uncertainty estimation. It does not need to be
deleted after the initial split. Do not print individual identifiers in reports.

## `admission_type_id`

`admission_type_id` is a nominal categorical variable that identifies the type of
hospital admission for an encounter. Its numeric codes do not represent quantities
or a meaningful order.

It is a candidate predictive feature. We may investigate its association with
recorded readmission and evaluate its contribution to prediction. An observed
association would not establish a causal effect.

The mapping source is `data/raw/IDS_mapping.csv`, section `admission_type_id`.

Three codes indicate that a known admission type is unavailable:

- `5` — Not Available
- `6` — NULL
- `8` — Not Mapped

In the original local dataset, these codes account for **10,396 of 101,766
encounters (10.22%)**, before any cohort selection or cleaning. Counts and
percentages were obtained using `examples/phase_1/inspect_csv.py`.

No transformation or row exclusion has been decided. The original codes remain
distinct and unchanged; combining their counts for this summary does not imply
that they are equivalent.

## `age`

`age` is an ordinal categorical variable that records the patient's age interval
for an encounter, rather than an exact age. 

Replacing each interval with its midpoint is a possible approximation. No encoding or
transformation has been decided yet.

It is a candidate predictive feature. We may investigate the association between
age groups and recorded readmission; this would not establish a causal effect.

## `A1Cresult`

`A1Cresult` records a categorical representation of the HbA1c test result or
indicates nonmeasurement. HbA1c reflects average blood glucose over approximately
the preceding three months. The exact numeric result is unavailable in this
column.

The original study documentation describes the categories as follows:

- `Norm`: HbA1c below 7%.
- `>7`: HbA1c above 7% and below 8%.
- `>8`: HbA1c above 8%.
- `None`: not measured.

The documentation does not explicitly specify category assignment at exactly 7%
or 8%. These historical labels must not be treated as universal clinical targets
or diagnoses.

The measured-result categories have a natural order, while `None` is a separate
nonmeasurement category outside that order.

In the local dataset, `None` occurs in **84,748 of 101,766 encounters (83.28%)**.
It is literal text, not Python `None`, and does not indicate a normal result.

This is a candidate predictive feature. No transformation has been decided yet.

Sources:

- [Original study, Table 1](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/7b693390-2ca7-46bc-953f-357ee1f06ec1): dataset category definitions.
- [NIDDK: The A1C Test & Diabetes](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test): test meaning and individualized clinical targets.
- `scripts/profile_data.py`: local category counts and percentages.

## Prediction-time availability

### Current extended-model missing-category policy (2026-11-05)

For the first broader model, plan to include race, gender, payer_code and
medical_specialty as nominal categories. Keep literal ? as missing-information
category and recorded Unknown/Invalid gender unchanged; do not impute an
observed class or a numerical zero. The one-hot vocabulary is fitted within
training folds. This represents recorded missingness without identifying its
mechanism or establishing causal demographic effects.

Exclude weight from this primary extension: the learner's saved-training audit
finds 58,365/60,276 unavailable values (96.83%). The remaining values are grouped
weight intervals, not exact measurements. This is a pragmatic initial scope
choice, not proof of irrelevance or a general missingness threshold. Retain the
original column and all encounters in source/cohort data. See D-040 for the
training-only counts and rationale. The initial six-feature reference is unchanged.

### Availability inventory

Current-model update (2026-11-20): `FEATURE_SETS["reviewed"]` excludes
`payer_code`, `acetohexamide`, `examide` and `citoglipton` from the 45-field
historical `extended` schema, leaving 41 predictors. The three drug fields
are constant in saved training; payer removal follows a bounded training-CV
review, not missingness alone. All original columns remain in source data;
the inventory below records source semantics and earlier candidate roles.
See [FEATURE_REVIEW.md](FEATURE_REVIEW.md) for evidence and retained fields.

Current primary-model choice (D-041): plan to include diag_1/diag_2/diag_3
under an explicit, unverified discharge-time availability assumption for this
retrospective study. Their final recorded codes may have been finalized later;
the dataset cannot resolve this. Preserve strings and separate diagnosis positions.
Omit discharge_disposition_id as a predictor in this first broader comparison
because of its documented timing ambiguity, without claiming the other fields'
timing is verified. Its use in cohort selection remains unchanged and uncertain
eligibility timing is still a study limitation. These are modeling choices, not
proof of prospective availability or leakage absence. Encoding implementation
and training-only cardinality checks are pending.

The prediction time is hospital discharge. The dataset contains retrospectively
collected records, without field-level recording timestamps or revision histories.
We therefore cannot fully verify whether each candidate feature was accessible
at prediction time. Availability assumptions and unresolved timing uncertainties
must be documented.

| Field | Role | Availability at discharge | Proposed predictive use | Rationale |
|---|---|---|---|---|
| `readmitted` | Target | No: determined from subsequent recorded events | Outcome only; never a predictor | Records `<30`, `>30`, or `NO`, rather than an exact number of days. `NO` denotes no recorded readmission. |
| `patient_nbr` | Identifier | Assumed available | Grouping for splits; excluded from predictors | Links encounters belonging to the same patient and supports patient-disjoint evaluation. |
| `number_inpatient` | Candidate feature | Assumed available; operational timing unverified | Candidate numeric predictor | Counts inpatient encounters in the preceding year. It describes prior utilization, without establishing that previous diagnoses were similar. |
| `diag_1` | Candidate feature | Uncertain: definitive coding time is unavailable | Candidate categorical predictor; inclusion requires an explicit availability assumption | Records the primary diagnosis code for the encounter. The dataset does not establish when the definitive code became accessible. |
| `encounter_id` | Encounter identifier | Assumed available | Metadata only; excluded from predictors | Uniquely identifies an encounter and supports joins, duplicate checks and source/import reconciliation. It is not a clinical measurement. |
| `time_in_hospital` | Candidate feature | Assumed available once the encounter is complete; recording time unverified | Candidate numeric predictor | Records the completed encounter's duration in days. The final duration would be future information for a prediction made at admission. |
| `num_medications` | Candidate feature | Assumed available by discharge; aggregation/recording time unverified | Candidate numeric predictor | Counts distinct generic medication names administered during the encounter, not necessarily the medications prescribed at discharge. |
| `discharge_disposition_id` | Nominal categorical field; cohort-selection variable and candidate feature | Not verified for all codes; event and recording times are unavailable | Used for the documented cohort rule; predictive inclusion remains undecided pending timing review | Describes discharge status or destination. Its codes identify categories, not individual encounters, and have no quantitative order. Some local death-related labels have unresolved event timing. |

### Availability by feature family

Together with the individual rows above, these families cover all 50 source
columns. They are working assessments for the retrospective discharge-time
study, not proof of real-time availability or a finalized model feature list.
Grouping fields does not make their types or meanings interchangeable.

| Family / fields | Role | Availability at discharge | Proposed predictive use | Rationale |
| --- | --- | --- | --- | --- |
| Demographics: `race`, `gender`, `age`, `weight` | Candidate features | Assumed obtainable by discharge; field-level recording times unverified | Nominal predictors for recorded `race`/`gender`; ordered interval categories for `age`/`weight`, with encoding undecided | These are recorded patient attributes. `age` and `weight` are grouped values in the local CSV, not exact measurements. Unknown or missing labels do not justify automatic row exclusion. |
| Admission/administrative context: `admission_type_id`, `admission_source_id`, `payer_code`, `medical_specialty` | Candidate features | Assumed available for this study by discharge; administrative finalization times unverified | Candidate nominal categorical predictors | Describe admission type, admission source, payer category and specialty of the admitting physician. Numeric category codes are not encounter identifiers or quantitative magnitudes. |
| Prior utilization: `number_outpatient`, `number_emergency` | Candidate features | Historical windows precede the encounter; availability/completeness of recorded history assumed | Candidate nonnegative integer predictors | Count outpatient and emergency visits, respectively, in the year preceding the encounter. They are not counts of this patient's rows in the CSV and do not establish similar diagnoses or complete external history. |
| Encounter counts: `num_lab_procedures`, `num_procedures`, `number_diagnoses` | Candidate features | Assumed available when encounter summaries are complete; aggregation/finalization times unverified | Candidate nonnegative integer predictors | Count laboratory tests, other procedures and diagnoses entered for the encounter. The diagnosis count is not limited to the three supplied diagnosis-code fields. Completed-encounter totals cannot be assumed available at admission. |
| Other diagnosis positions: `diag_2`, `diag_3` | Candidate features | Uncertain: definitive coding times are unavailable | Candidate categorical predictors; inclusion requires an explicit availability assumption | Record secondary and additional secondary diagnosis codes. Field position does not establish severity or the time at which the code became accessible. Preserve code strings rather than treating them as numeric quantities. |
| Laboratory categories: `max_glu_serum`, `A1Cresult` | Candidate features | Measured results assumed accessible by discharge for this study; result posting and nonmeasurement finalization times unverified | Candidate categorical predictors, retaining nonmeasurement as a separate source category; encoding undecided | Record result ranges or literal `None` for not measured. Frequent nonmeasurement does not by itself make a feature unusable; predictive usefulness has not been established in this project. These are not exact numeric laboratory results. |
| All 23 drug-specific columns, from `metformin` through `metformin-pioglitazone` in the verified source order | Candidate features, including two observed constant columns | Summarize prescribing/dosage during the encounter; assumed available by discharge, with finalization times unverified | Candidate categorical predictors; any constant-feature filtering belongs to training-fitted preprocessing | `No` denotes not prescribed; `Steady`, `Up` and `Down` describe dosage status/change during the encounter. They do not reconstruct pre-admission therapy or the discharge prescription. `examide` and `citoglipton` are constant at `No` in the source audit; no columns have been removed. |
| Medication indicators: `change`, `diabetesMed` | Candidate features | Assumed available by discharge as encounter summaries; recording times unverified | Candidate nominal categorical predictors with separate allowed domains | `change` indicates a change in diabetic medication dosage or generic name (`No`/`Ch`). `diabetesMed` indicates any diabetic medication prescribed (`No`/`Yes`). `change = No` does not imply `diabetesMed = No`. |

The drug-column range refers to the exact source schema already checked during
ingestion and audit; it excludes `change` and `diabetesMed`, listed separately.
Source values remain unchanged. Availability uncertainty is a limitation to
document, not evidence that every uncertain field contains future information.

The source audit found literal `None` in 96,420 `max_glu_serum` records (94.75%) and
84,748 `A1Cresult` records (83.28%). These are nonmeasurement frequencies, distinct
from question-mark missingness. Nonmeasurement may be informative about the
measurement process, but any predictive contribution or generalizability remains
to be evaluated. The literature context below motivates hypotheses, without
establishing predictive performance in CarePath Lab or automatically excluding
either laboratory field.

### Literature context for laboratory candidates

Reviewed on 2026-10-31 after the learner supplied three sources. Findings below
are attributed to their authors; no model or result has been reproduced here.

- **Madushani, Pathiravasan and LaRosa, progress report (2018), pp. 1-2:**
  [Predicting Hospital Readmission for Diabetes Patients](https://www.soihub.org/site/assets/files/6113/diabetesprobability-midyear-progress-report.pdf).
  Reports relatively low HbA1c-related importance in its random forests and a
  reduced GLM omitting HbA1c/glucose-related predictors. This supports examining
  their incremental value in those settings, not universal irrelevance. Some
  forests combine HbA1c with medication change; later models separate them. This
  is a progress report, and patient-disjoint evaluation is not established for
  the repeated-encounter forest from the supplied two-page description. Random
  forests do not remove the need to check patient overlap during evaluation.
- **Meshref, IJCA (2023), sections 3.3 and 5, pp. 22 and 25-26:**
  [Investigating the Impact of Prominent Factors on the Diagnosis of Diabetes and its Associated Diseases using Ensemble Machine Learning Models](https://www.ijcaonline.org/archives/volume184/number48/meshref-2023-ijca-922598.pdf).
  Uses the source dataset for different diagnostic targets. In its disease
  classification analysis, it removes `max_glu_serum` because of frequent
  nonmeasurement in the reduced data, while retaining `A1Cresult`. This is a
  task-specific preprocessing choice, not evidence that both fields are unusable
  for early-readmission prediction.
- **Muñoz Guerrero, AJBR, labeled 2015, local supplied PDF:**
  [Publisher entry and DOI](https://africanjournalofbiomedicalresearch.com/index.php/AJBR/article/view/10053).
  Tables 4-5 report no statistically significant adjusted HbA1c-category
  associations and a weak exposure-only model in a first-encounter cohort of
  69,987 patients excluding death/hospice discharges. Our repeated-encounter,
  hospice-retaining cohort differs. These findings do not prove zero predictive
  contribution. **Bibliographic caveat:** the entry/PDF say 2015, but PDF p. 3
  states Python 3.12, whose [initial release was 2023-10-02](https://www.python.org/downloads/release/python-3120/).
  The version/publication chronology needs clarification; do not treat this
  document as settled evidence for a design decision. No conclusion about the
  cause of this inconsistency is asserted.

Statistical significance, model-specific importance and out-of-sample incremental
prediction are different questions. These sources justify documented comparisons
with and without laboratory fields in future training/validation experiments,
not selection using the held-out test set. High nonmeasurement alone remains
insufficient to declare a feature unusable. No feature has been dropped.

The feature list, encodings, imputations and any data-dependent selection will be
specified in Phase 2 using training data and the designated validation procedure,
without fitting on or choosing features from held-out test results. Target and
identifiers remain excluded from predictors by design.

Field definitions: [UCI dataset documentation](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
and [original study, Table 1](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/7b693390-2ca7-46bc-953f-357ee1f06ec1).
The availability assessments above are declared assumptions or uncertainties,
not source-verified operational timestamps. Candidate status is not a final
decision to include a field in the baseline model.
