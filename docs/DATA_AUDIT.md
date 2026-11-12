# Preliminary data audit

Last updated: 2026-10-31. These are initial descriptive observations, not a complete
data-quality audit or model evaluation.

## Full-source import verification

On 2026-10-29 the learner supplied live output from
`scripts/verify_raw_import.py`: 101,766 source encounters and 101,766 database
encounters, matching ordered column names and zero missing, unexpected or
mismatched records. The verifier compares all 50 parsed field values by
encounter_id, independently of row order. The inspected `encounters_raw` snapshot
therefore matches the local CSV values exactly. This establishes import fidelity,
not clinical accuracy, source authenticity or byte-for-byte CSV equivalence.

On 2026-10-31 the tutor independently reran the live PostgreSQL verifier with the
same successful result. The four source inspection/audit scripts and all 33
saved synthetic tests passed. Original source fingerprints still match README;
regenerating the marker report produced identical bytes. These checks close the
initial Phase 1 scope; they are not model evaluation or an exhaustive clinical
validation. Automated tests mock database connections; the live comparison is
a separately executed acceptance check.

Repository cleanup later on the same date renamed the broad profiler to
`scripts/profile_data.py` and moved the original CSV exercises to
`examples/phase_1/`. The profiler, administrative audit and learner inspector
produce identical output to their pre-cleanup versions; the aggregate marker
report and original source fingerprints are unchanged. Shared validation/cohort
functions are now in `carepath_lab`. Historical checks above keep their original
test count; the consolidated suite now has 34 passing tests and the post-cleanup
live verifier again reports zero discrepancies. Current commands and detailed
acceptance results are recorded in README and the current `docs/PHASE_2_REPORT.md`;
the local mentoring handoff is maintained separately.

## Medication categories

On 2026-10-30 four source distributions were independently checked; each totals
101,766 encounters:

| Field | Observed category counts |
| --- | --- |
| metformin | No: 81,778; Steady: 18,346; Up: 1,067; Down: 575 |
| insulin | No: 47,383; Steady: 30,849; Down: 12,218; Up: 11,316 |
| change | No: 54,755; Ch: 47,011 |
| diabetesMed | Yes: 78,363; No: 23,403 |

According to [original study Table 1](https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/7b693390-2ca7-46bc-953f-357ee1f06ec1),
drug-specific No denotes not prescribed, not missing. Steady denotes no dosage
change during the encounter; it does not establish the pre-admission regimen.
The aggregate change field concerns diabetic medication changes (dosage or
generic name), whereas diabetesMed indicates any diabetic medication prescribed.
Interpret each label in its own field. No encoding or exclusion has been chosen.

The subsequent drug-specific audit covers all 23 columns from metformin through
metformin-pioglitazone in the verified source order. All observed labels belong
to {No, Steady, Up, Down}; no unexpected labels were found. Examide and citoglipton
are constant at No for all 101,766 encounters. This is a sample-level observation,
not evidence about prescribing outside this dataset. No columns were removed;
any fitted feature-selection rule must later use the training partition only.

Explicit observed-minus-allowed checks also found no unexpected categories in
change ({No, Ch}) or diabetesMed ({No, Yes}). These checks validate each field's
observed labels separately; they do not establish cross-field consistency.

## Diagnosis-code profile

On 2026-10-30 the source profile excluded literal ? only from this summary,
preserving the original strings and rows:

| Field | Distinct recorded codes | Encounters with a recorded code | ? markers |
| --- | ---: | ---: | ---: |
| diag_1 | 716 | 101,745 | 21 |
| diag_2 | 748 | 101,408 | 358 |
| diag_3 | 789 | 100,343 | 1,423 |

For each field, full frequency totals plus marker counts equal 101,766. The
five most frequent codes and counts printed by the learner were independently
confirmed on the local CSV. These are observed cardinalities, not an external
ICD-9 validity check or expected cardinality constraints. Codes remain categorical
strings; no truncation, grouping or numeric encoding was performed.

## Administrative mapping coverage

On 2026-10-30 all three administrative fields were compared with their own
section keys in IDS_mapping.csv:

| Field | Observed distinct codes | Mapping entries | Unmapped observed codes |
| --- | ---: | ---: | ---: |
| admission_type_id | 8 | 8 | 0 |
| discharge_disposition_id | 26 | 30 | 0 |
| admission_source_id | 17 | 25 | 0 |

Unused mapping entries are not a coverage error. Complete observed coverage
does not establish correct code assignment or complete information; descriptions
such as NULL and Not Mapped remain explicit source labels. No source values were
changed.

Descriptions in {Not Available, NULL, Not Mapped, Unknown/Invalid} were grouped
only for this audit, using strip for comparison without altering stored labels:

| Field | Encounters with unavailable information | Percentage of all 101,766 encounters |
| --- | ---: | ---: |
| admission_type_id | 10,396 | 10.22% |
| discharge_disposition_id | 4,680 | 4.60% |
| admission_source_id | 7,067 | 6.94% |

The counts were independently confirmed from the source on 2026-10-30. They count
encounters, not distinct codes. A single encounter can contribute to multiple
fields, so their sum is not a count of unique affected encounters. Complete
mapping coverage coexists with unavailable information. No codes were merged or
source records removed.

## Candidate duplicate records

On 2026-10-30 the source audit found zero rows matching another row on all 49
fields other than encounter_id. The comparison retains patient_nbr and readmitted;
`duplicated(subset=comparison_columns, keep=False)` counts every row participating
in an exact-match group. No records were removed. This result does not rule out
repeated clinical events with differing recorded values, partial matches or near
duplicates. Even an exact match would require investigation before deletion.

## Recorded early readmission by age group

In the original local dataset, the observed proportion of encounters with
`readmitted == "<30"` is 14.24% in age group `[20-30)` and 11.77% in `[70-80)`.
Each denominator includes all encounters in the corresponding age group,
including outcomes `>30` and `NO`.

| Age interval | Encounters with outcome `<30` / all encounters in group | Percentage |
| --- | ---: | ---: |
| `[20-30)` | 236 / 1,657 | 14.24% |
| `[70-80)` | 3,069 / 26,068 | 11.77% |

These two groups illustrate why a larger number of encounters does not imply a
higher proportion of recorded early readmissions. They are selected descriptive
examples, not a complete comparison of all age groups.

The `[20-30)` group has fewer encounters. With fewer independent observations,
an estimated proportion is generally more sensitive to sampling variability.
However, uncertainty has not yet been quantified here, and encounters from the
same patient may be dependent. Encounter counts are not counts of distinct
patients, and a larger sample does not remove selection bias.

These unadjusted frequencies do not establish that age causes readmission or
that the difference is statistically significant. Other patient and encounter
characteristics may differ between groups. Conclusions are limited to the
recorded outcomes in this selected historical dataset (1999–2008), before any
cohort selection or cleaning; they do not establish current clinical risk.

Source: `data/raw/diabetic_data.csv`. Counts and percentages are produced by
`examples/phase_1/inspect_csv.py`. Run from the project directory:

```bash
.venv/bin/python -m examples.phase_1.inspect_csv
```

## Patients with multiple recorded encounters

The original local dataset contains **101,766 encounters from 71,518 distinct
patient identifiers**. Of these patients, **16,773 (23.45%)** have at least two
recorded encounters. The denominator is the number of distinct patients, not
the number of encounters. These aggregates are produced by the same script.

Multiple recorded encounters do not necessarily indicate readmission within
30 days. They also do not describe a patient's complete hospitalization history.
The 30,248 encounters beyond one per patient are not a count of patients with
multiple encounters. Repeated patients motivate patient-grouped model splits
and must be considered when quantifying uncertainty.

## Admission-type mapping coverage and information availability

Every admission-type code observed in `encounters_staging` has a matching entry
in `admission_type_mapping`: the LEFT JOIN check found **zero encounters without
a mapping match**. This establishes mapping coverage for the observed codes,
not the accuracy of the original admission coding.

However, codes `5` (Not Available), `6` (NULL) and `8` (Not Mapped) do not specify
a known admission type. Together, they occur in **10,396 of 101,766 encounters
(10.22%)**. The description `NULL` is literal text, not a SQL NULL value.
Mapping coverage is complete for the observed codes, while admission-type
information remains incomplete. No recoding or row exclusion has been decided.

The learner verified mapping coverage using `examples/phase_1/sql/008_unmapped_admission_types.sql`
and mapped encounter counts using `examples/phase_1/sql/007_encounters_by_admission_type.sql`.
The combined frequency of codes 5, 6 and 8 was calculated in
`examples/phase_1/inspect_csv.py`.

## Hospital stay duration

Inspection of all 101,766 original encounters found no nonnumeric
`time_in_hospital` values, no fractional durations and no values outside 1–14
days. The observed minimum is 1 and maximum is 14, consistent with the source's
documented encounter inclusion criteria. Checks are implemented in
`scripts/profile_data.py`, using a separate parsed Series without
modifying the source strings.

These checks establish numeric format and range consistency, not accuracy of
each recorded duration. They do not justify generalizing results to hospital
stays outside the dataset's inclusion range.

The [official UCI description](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
explicitly states that encounters were selected with stays of at least 1 and
at most 14 days. Thus 14 is a documented dataset inclusion bound, not a universal
maximum hospital stay. A source value of 15 would require investigation against
this contract; it would not be clinically impossible. In contrast, observed
maxima for procedure, medication and prior-visit counts are descriptive findings,
not justified upper validation bounds. Do not clip or reject values solely
because they exceed those observed maxima.

## Procedure and medication counts

The following columns contain no nonnumeric, negative or fractional values in
the original local dataset:

| Column | Observed minimum | Observed maximum |
| --- | ---: | ---: |
| `num_lab_procedures` | 1 | 132 |
| `num_procedures` | 0 | 6 |
| `num_medications` | 1 | 81 |

These are observed extrema, not predefined validity thresholds or evidence that
each source value is accurate. The audit uses separately parsed Series and
counts conversion failures separately from fractional values. The numeric audit
is printed by `scripts/profile_data.py`; it is not included in the
missing-marker CSV export.

## Diagnosis count

On 2026-10-30 the number_diagnoses audit found observed minimum 1 and maximum 16,
with zero numeric parsing failures, fractional values or negatives across all
101,766 source records. The observed maximum is not a justified hard upper bound.
This count is not restricted to the three supplied diag_* code positions; these
checks establish numeric/domain consistency, not diagnosis accuracy. Source
values were not modified.

## Prior healthcare utilization counts

The learner extended the numeric audit to recorded visits in the year preceding
each encounter. There are no nonnumeric, fractional or negative values in the
three fields:

| Column | Observed minimum | Observed maximum |
| --- | ---: | ---: |
| `number_outpatient` | 0 | 42 |
| `number_emergency` | 0 | 76 |
| `number_inpatient` | 0 | 21 |

Zero refers to the source's preceding-year observation window, not an absence
of lifetime healthcare use. Extrema alone do not establish errors, clinical
plausibility or the frequency of high counts. No upper-bound exclusion is imposed.

## Discharge disposition and recorded readmission

The learner supplied the following cross-tabulation from the original CSV:

| Discharge code | `<30` | `>30` | `NO` |
| --- | ---: | ---: | ---: |
| `11` — Expired | 0 | 0 | 1,642 |
| `19` — Expired at home | 0 | 0 | 8 |
| `20` — Expired in a medical facility | 0 | 0 | 2 |
| `21` — Expired, place unknown | 0 | 0 | 0 |
| `13` — Hospice / home | 19 | 36 | 344 |
| `14` — Hospice / medical facility | 24 | 7 | 341 |
| `2` — Transfer to another short-term hospital | 342 | 668 | 1,118 |

Code 21 is absent from observed records; its zero row was added for display.
Hospice is not equivalent to death. `NO` denotes no recorded readmission and
does not establish survival or complete outcome ascertainment. Discharge-code
semantics and timing must motivate any cohort exclusions; observed target
frequencies alone must not determine inclusion. No exclusion has been applied
to source files or the staging table. Timing of codes 19/20/21 requires review
before treating them as equivalent to code 11 for a discharge-time prediction.

## Provisional cohort: excluding discharge code 11

A separate in-memory selection excludes code `11` (Expired), motivated by the
candidate research population of encounters ending in discharge alive. This is
not a finalized cohort rule: other death-related codes and their timing remain
under review, so the remaining records cannot yet be labeled all discharged alive.

| Selection | Encounters | Recorded `<30` outcomes | Prevalence |
| --- | ---: | ---: | ---: |
| Original dataset | 101,766 | 11,357 | 11.16% |
| Excluding code 11 only | 100,124 | 11,357 | 11.34% |
| Excluding codes 11, 19, 20 and 21 | 100,114 | 11,357 | 11.34% |

All 1,642 excluded encounters have target `NO`. Positives are unchanged; the
prevalence increases because the denominator decreases. This is a change in
population composition, not an improvement in predictive performance. Original
CSV files and the PostgreSQL staging table have not been filtered by this exercise.

The second candidate excludes 10 additional encounters (8 with code 19 and 2
with code 20; code 21 is absent). Its prevalence is 11.34407%, compared with
11.34293% for exclusion of code 11 alone; both round to 11.34%. This sensitivity
comparison does not resolve code timing or establish the final cohort. The rule
covers the selected death-related labels, not independently verified complete
mortality ascertainment.

### Working-rule correction after learner review

The working analysis now excludes all four local death-related codes
`11`, `19`, `20` and `21`. Retaining 19/20/21 solely because timing was uncertain
was not sufficiently justified: place of death does not establish eligibility.
The rule remains based on source labels with unverified event dates and incomplete
mortality ascertainment. The code-11-only cohort above is a historical diagnostic
comparison, not the current primary selection. See `COHORT_POLICY.md`.
