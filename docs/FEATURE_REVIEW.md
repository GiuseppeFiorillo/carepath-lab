# Predictor review — 2026-11-20

The learner requested a source-informed review and authorized bounded experiments,
actual predictor removal where justified, and refreshed current-model artifacts.
The frozen 45-predictor Phase 2 results remain historical evidence; this is a
new exploratory model revision after repeated inspection of development validation.

## Evidence and scope

There is no universal deletion list for this dataset. The original
[Strack et al. study, sections 2.2–2.3](https://doi.org/10.1155/2014/781670)
omits weight and payer code but retains missing specialty as a category.
[Goudjerkan and Jayabalan, 2019, section III](https://thesai.org/Downloads/Volume10No2/Paper_36-Predicting_30_Day_Hospital_Readmission_for_Diabetes_Patients.pdf)
removes constant examide/citoglipton but explicitly retains payer and specialty.
The [Fairlearn tutorial preparation](https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py)
uses a different 25-column representation, retaining laboratory nonmeasurement,
specialty and demographics, and deriving payer indicators. These sources motivate
candidates, not a claim that their performance estimates or cohort rules transfer
to our patient-disjoint problem. We do not copy fixed missingness cutoffs,
oversampling or preprocessing performed before validation splits.

## Training-only inventory

Only saved training data are used for this review: 60,276 encounters and 42,263
patients. Sources and saved assignments are fingerprinted. No test predictions,
scores or test-driven column decisions are used.

| Candidate | Training evidence | Review rule |
| --- | --- | --- |
| `encounter_id`, `patient_nbr`, `readmitted` | Identifiers and target | Already excluded from X; retained for integrity/grouping/target |
| `weight` | 58,365 `?` (96.83%) | Keep previous pragmatic exclusion; no source-row deletion |
| `discharge_disposition_id` | Already excluded from X by project policy | Keep previous scope; at-discharge availability alone does not make every discharge field leakage |
| `examide`, `citoglipton`, `acetohexamide` | Each constant in saved training | Remove from reviewed X; do not claim all three constant in unseen populations |
| `payer_code` | 23,569 `?` (39.10%) | Test removal; administrative/missingness associations may be fragile |
| `medical_specialty` | 29,553 `?` (49.03%) | Test removal; missingness alone does not establish irrelevance |
| `max_glu_serum`, `A1Cresult` | 57,110 and 50,270 literal `None` | Test joint removal; `None` is recorded nonmeasurement, not a normal result |
| `diag_1`–`diag_3` | High-cardinality clinical categories | Test removal, not automatic exclusion; no timestamp proof of availability |
| `race`, `gender` | Recorded demographic categories | Test predictor removal; retain source columns for subgroup audits regardless |
| Rare nonconstant medication categories | Some drugs observed in fewer than 20 training patients | Rarity alone is insufficient for blanket removal; no new frequency-threshold search |

## Bounded protocol declared before CV results

Eight profiles: 45-field reference, three-constant removal, and constant-cleaned
versions without payer, specialty, both payer/specialty, both labs, all diagnosis
codes, or race/gender. Configurations exist only within the review process; they
do not create a permanent feature set for every ablation.

All use the same five shuffled patient GroupKFold splits, seed42, on saved
training; a fresh model/preprocessor is fitted within each fold. CatBoost remains
200 iterations, depth6, learning rate .05, L2=3, seed42, eight threads. Specialty
mode imputation is fitted in the fold only if specialty is retained. No new
hyperparameter tuning or saved-validation selection.

Always omit verified constants from the revised schema. Additional removal can
be preferred for simplicity/availability if mean held-out AP decreases by no
more than .002 and mean Brier increases by no more than .0002 relative to the
constant-cleaned reference. These are explicit exploratory engineering
tolerances, not clinically validated or statistically demonstrated equivalence
margins. AUROC and foldwise changes are secondary descriptive checks. Verify
the combined removal once before adopting it; do not iteratively search subsets.

## Results and decision

All 40 evaluations completed. Mean held-out training-CV scores:

| Profile | Predictors | AP | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: |
| Historical reference | 45 | 0.212184 | 0.652371 | 0.097889 |
| Remove training constants | 42 | 0.211182 | 0.651834 | 0.097931 |
| Remove constants + payer **(selected)** | 41 | 0.211761 | 0.650868 | 0.097941 |
| Remove constants + specialty | 41 | 0.210700 | 0.650566 | 0.097970 |
| Remove constants + payer + specialty | 40 | 0.210216 | 0.650053 | 0.098000 |
| Remove constants + laboratory categories | 40 | 0.211666 | 0.652108 | 0.097913 |
| Remove constants + diagnosis codes | 39 | 0.204520 | 0.639721 | 0.098326 |
| Remove constants + race/gender | 40 | 0.211405 | 0.650619 | 0.097951 |

Full precision: [aggregate CV table](reports/feature_review_cv.csv). These are
exploratory comparisons after earlier model development, not equivalence tests
or a new confirmatory benchmark. Removing constant columns can still change
finite CatBoost fits through feature-index/randomization choices; constant
training values themselves supply no learnable between-record contrast.

**Remove from the current predictor schema:** `payer_code`, `acetohexamide`,
`examide`, `citoglipton`. No raw column, encounter or patient is deleted.
The combined four-column removal is already the evaluated no_payer profile,
so a further identical joint CV run is unnecessary. Relative to no_constants,
mean AP changes by +0.000579, AUROC by -0.000966 and Brier by +0.000010.
Relative to the historical 45-field model, AP changes by -0.000422. This supports
a modest simplicity/administrative-dependence tradeoff, not a claim of better
performance, demonstrated equivalence or improved fairness. Payer's 39.10%
missingness is contextual evidence, not a standalone deletion threshold.

**Retain:**

- `medical_specialty`: its removal reduces AP in four of five folds and AUROC
  in all five; Brier worsens in all five. The effect is small but there is no
  compelling reason to discard this recorded clinical context solely because
  many entries are missing.
- `max_glu_serum`, `A1Cresult`: removal has mixed fold effects (AP improves in
  two folds and declines in three). Neither the small aggregate gain nor
  frequent `None` establishes irrelevance. Retain recorded measurement and
  nonmeasurement information rather than expanding a deletion search.
- `diag_1`–`diag_3`: removal worsens all three metrics in every fold; mean AP
  loss .006662 and Brier increase .000395 exceed the declared tolerances.
  High cardinality is not a sufficient reason to delete them. Discharge-time
  availability remains an assumption, not verified timestamps.
- `race`, `gender`: AP changes are mixed and AUROC declines in every fold.
  Removal is not established as a fairness remedy. Retain for the current
  exploratory model and source subgroup audit; appropriateness for a real
  intervention would need a separate contextual decision.
- Remaining nonconstant medications: do not remove all low-frequency drugs
  merely for rarity. Support limitations belong in the interpretation.

Meeting a score tolerance permits a reasoned removal; it does not compel
deleting every such group or prove that combining individually harmless
removals is harmless. We adopt only the narrowly motivated payer/constant
removal and stop this review rather than iterating through subsets.

## Current model and historical compatibility

`FEATURE_SETS["reviewed"]` is the current 41-predictor schema (8 numeric,
33 categorical). `initial` (6) and `extended` (45) remain unchanged historical
references; their earlier metrics and saved pipelines must not be relabeled.
The selected model remains CatBoost with the same hyperparameters and
training-fitted specialty mode imputation, no scaling or missingness indicator.

Detailed profiles, full-precision fold scores, protocol, selection, hashes and
environment are under ignored `artifacts/feature_review_2026-11-20/`.
After this training-only decision, fit on all saved training and export a new
pipeline/predictions under `artifacts/feature_review_selected_final_2026-11-20/`.
The descriptive development-validation scores are AP 0.205039, AUROC 0.650958,
Brier 0.096123; historical 45-field values were 0.205999/0.653226/0.096056.
These development scores do not determine the feature decision and do not
show improvement. No final-test predictions or metrics were computed.

New SHAP explanations must use the new pipeline and `feature_set="reviewed"`.
The previous 45-feature local explanations describe only the historical model.
The refreshed five-encounter native SHAP smoke check verifies raw-score
additivity; a global explanation and full fairness report remain pending.

The [selection record](reports/feature_review_selection.json) and
[all 40 aggregate fold results](reports/feature_review_folds.csv) are versioned,
so refitting the frozen reviewed model does not require reconstructing a local
decision JSON. Refresh verifies the fold-file digest against that record.
Current audit outputs are under
`artifacts/feature_review_audit_final_2026-11-20/`; the source can be chosen
explicitly with the audit command's `--source-artifacts` argument.
The refreshed audit retains all 19,883 validation encounters/2,233 positives;
its calibration table has 92 nonempty subgroup/bin combinations. Recall
Female-minus-Male at .20 is -1.33 percentage points with a 95% patient-cluster
percentile interval [-6.61,+3.62] (1,000/1,000 finite differences). This is not
a test that removing payer improved fairness, nor a simultaneous guarantee.
