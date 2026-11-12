# First-baseline cohort and target policy

Date: 2026-10-31. Status: initial baseline contract, consolidated from the audit
and learner review. This supersedes the earlier code-11-only proposal. New
criteria would require an explicit revision; none are added by this consolidation.

## Research population and prediction time

Study recorded early readmission among eligible encounters in the historical
UCI dataset, using information available by discharge. Retain repeated patients
and separate patients across training, validation and test partitions. Do not
claim complete follow-up, current clinical utility or verified discharge-alive
status for every included encounter.

## Working primary analysis

- Exclude local discharge codes `11`, `19`, `20` and `21`, all of which describe
  death in the provided mapping. Location of death is not a reason to retain a
  death-coded record. This leaves 100,114 encounters and 11,357 recorded `<30`
  outcomes (11.34%). The rule is based on recorded labels, not verified death dates.
- Retain hospice codes `13`/`14`: hospice is not death; excluding hospice would
  define a narrower research population and requires a separate rationale.
- Retain transfer codes initially. Report the limitation that the source does
  not let us reliably reconstruct continuous care episodes or validate every
  transfer/readmission distinction.
- Do not exclude unknown categories or missing features solely to improve model
  metrics. Additional inclusion rules require documented reasons and impact counts.

## First-baseline contract

### Unit of analysis and partitions

One row represents one encounter, not one unique patient. Retain repeated
encounters as separate observations rather than aggregating them or selecting
only one encounter per patient. All encounters of a patient must stay in a single
training, validation or test partition. Keep `patient_nbr` for grouping and verify
patient disjointness; any later cross-validation must also respect patient groups.
No model split has been implemented yet.

### Source preservation and eligibility

Apply the four-code rule to `discharge_disposition_id`, which describes discharge
status/destination; these are not `readmitted` outcome codes. Keep original CSV
files and `encounters_raw` complete. The analysis cohort is a separate derived
selection. For this first baseline, retain other eligible records, including
hospice/transfer records, unknown feature categories and nonmeasurement labels.
There are no additional row exclusions in this contract.

### Fixed recorded early-readmission target

The baseline predicts the recorded `<30` outcome, not readmission at any later
time. Validate source categories before mapping them:

| Recorded category | Binary target | Meaning for this task | Encounters in the cohort |
| --- | ---: | --- | ---: |
| `<30` | `1` | Recorded early readmission | 11,357 |
| `>30` | `0` | Recorded later readmission; not positive for this target | 35,545 |
| `NO` | `0` | No recorded readmission; not positive for this target | 53,212 |

This gives 100,114 encounters, 11,357 positives (11.34%) and 88,757 negatives.
`NO` does not prove complete follow-up, absence of readmission outside the source
or zero future risk. We use recorded categories; exact event dates and the
classification of an event on exactly day 30 cannot be verified here.

An unexpected outcome category is a validation error. Raise an exception; do not
silently drop the row, impute its outcome or map it to `0`. During transactional
loading such an error must abort the operation rather than commit a partial load.
The binary mapping above is a fixed task definition, not learned preprocessing.
The raw database retains its original three-category outcome.

### Identifiers and predictors

- `readmitted` is used only to construct the outcome; exclude it and any derived
  target information from predictors.
- `patient_nbr` supports patient grouping and disjointness checks, including
  cross-validation. It is never a predictive feature.
- `encounter_id` supports encounter identity, uniqueness checks, joins and
  source/import reconciliation. It is never a predictive feature.
- Other columns remain candidates subject to the availability review in
  `DATA_DICTIONARY.md`. Cohort-selection use of discharge disposition does not
  automatically approve its predictive use; that timing decision remains explicit.

### Cleaning and learned preprocessing

Keep source representations unchanged during this phase. Treat `?`, literal
`None`, unknown-information codes and genuine empty cells according to their
field definitions rather than applying a global replacement. In particular,
laboratory `None` denotes nonmeasurement, not a normal result or an automatically
imputed numeric value.

In Phase 2, establish patient-grouped partitions before fitting learned
preprocessing. Fit imputation statistics, scaling parameters, learned category
encodings and constant-column filters on training data only. Apply those fitted
transformations to validation and test data without refitting. Within
cross-validation, fit on each training fold separately. Use validation for model
or feature choices and preserve test isolation. The specific methods and final
feature list have not yet been chosen or fitted.

## Diagnostic comparison

The earlier code-11-only candidate leaves 100,124 encounters and the same 11,357
positives. Retain this as an audit comparison showing the impact of ten additional
death-coded records, not the primary analysis or evidence that those records are
eligible. The target frequencies did not motivate the working rule. No model
comparison is implemented by this diagnostic exercise.

## Source review and uncertainty

The [original study, section 2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/)
selected the first encounter per patient and excluded death/hospice encounters.
CarePath Lab is a different predictive study, not an exact replication; those
choices are context, not automatic requirements.

`data/raw/IDS_mapping.csv` remains the source for local code meanings. The
[CMS discharge-status dictionary](https://bluebutton.cms.gov/resources/variables/ptnt_dschrg_stus_cd/)
uses different numeric codes. It does not supply a validated crosswalk or event
timestamps for UCI codes; do not infer such a crosswalk from similar labels.

The exact event timing remains unverified; exclusion of the four labels is an
explicit operational assumption, not proof of complete mortality ascertainment.
For this initial baseline, the eligibility rules above are the declared scope;
these uncertainties remain limitations rather than additional implicit filters.
Source files and PostgreSQL raw/staging tables remain unchanged. Do not use an
uncertain-timing status as a predictive feature without a separate availability
review.
