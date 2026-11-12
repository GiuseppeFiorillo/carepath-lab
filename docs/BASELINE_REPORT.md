# CarePath Lab — preliminary baseline comparisons

Historical development record. The completed Phase 2 procedures, closing
evaluation, figures and limits are in [PHASE_2_REPORT.md](PHASE_2_REPORT.md).

**Initial verified snapshot:** 2026-11-04; comparison updates through 2026-11-12  
**Status:** Preliminary report; three-model CV, saved-validation metrics/calibration, paired Brier intervals and RF-versus-LightGBM AP interval recorded. Final interpretation, result exports/figures and publication-readiness review remain pending.  
**Purpose:** Educational retrospective study of recorded early readmission.

## Imputation search closed for Phase 2 — 2026-11-12

At the learner's request to finish the phase, reviewer executed two final
five-fold training-only CatBoost profiles with frozen parameters. The first
mode-imputes race/payer_code/medical_specialty; the second additionally imputes
diag_1/diag_2/diag_3. Only literal "?" in these six selected predictors is treated
as missing. Lab "None" categories, unavailable administrative codes, numerical
predictors, identifiers, cohort and folds retain their documented treatment.
Every selected column has observed values in every fit fold.

| Profile | Mean AP | Mean AUROC | Mean Brier |
| --- | ---: | ---: | ---: |
| Specialty mode | 0.212183594 | 0.652370666 | 0.097889429 |
| Race mode | 0.211646513 | 0.652530805 | 0.097911462 |
| Specialty mode + indicator | 0.211421133 | 0.651866781 | 0.097921668 |
| Race/payer/specialty mode | 0.211301306 | 0.651715589 | 0.097921013 |
| Native explicit missing categories | 0.211287986 | 0.652031353 | 0.097936577 |
| All six question-mark predictors mode | 0.211012078 | 0.651644091 | 0.097928062 |
| Payer mode | 0.210254744 | 0.651263762 | 0.097970337 |

The new profiles took 43.2 and 43.7 seconds, respectively, on this environment.
Existing profiles are learner-executed references, read from full-precision CSVs;
new profiles are reviewer-executed. Results, paired per-fold differences and a
manifest with model/environment/protocol/input/code/output fingerprints are
saved under artifacts/catboost_final_imputation_2026-11-12. Input fingerprints
matched before/after fitting and new scores passed CSV readback checks. Current
code/environment snapshots do not retrospectively certify earlier reference
runs' environments; that provenance limitation is explicit in the manifest.

Select specialty-only mode without indicator by highest observed mean AP, keeping
the original CatBoost parameters fixed. This is a selected exploratory CV result,
not unbiased final performance or demonstrated superiority. Bigger imputation
groups did not improve observed mean AP. No imputed category is asserted to be
the patient's true unrecorded value. Stop model/imputation search for this phase.

No development-validation or final-test predictions in this execution block.
Remaining closure deliverables are fixed-procedure development-validation
evaluation with calibration/paired uncertainty and exported PR/calibration
figures, final comparative report and publication-readiness review. Existing
validation is repeatedly inspected development data. Phase 2 is not yet marked
complete, and final test remains reserved. No code changed for these fits, so
the synthetic suite was not rerun; last full suite: 105 passing.

## Additional native CatBoost single-column mode experiments — 2026-11-12

Reviewer reads the learner-saved race_mode.csv and payer_mode.csv (five folds,
eight columns each) and computes paired differences against the saved native
non-imputed reference. No independent reviewer refit. Parameters/features/folds
are the fixed D-052 protocol.

| Mode-imputed column | Mean AP | Mean AUROC | Mean Brier | Mean delta AP | Mean delta AUROC | Mean delta Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `race` | 0.211646513 | 0.652530805 | 0.097911462 | +0.000358527 | +0.000499452 | -0.000025114 |
| `payer_code` | 0.210254744 | 0.651263762 | 0.097970337 | -0.001033241 | -0.000767591 | +0.000033760 |

Race has favorable changes in three folds for each metric; payer AP worsens in
all five, while AUROC/Brier worsen in four. These remain descriptive comparisons
on overlapping training-CV fits, with no significance/equivalence claim or proof
about the missingness mechanism. Keep payer as a predictor; do not equate its
mode-imputation result with feature irrelevance. Next evaluate preservation of
missingness through a specialty-mode-plus-indicator candidate (D-053). Existing
CSV files are preserved; metadata consolidation remains pending.

## Training-only missing-marker/mode audit — 2026-11-12

Learner computes the following on the saved 60,276 training encounters. Mode
shares exclude the "?" rows and describe recorded values only, not correctness
probabilities for imputed values or target prevalence.

| Column | Question-mark count | Percentage of training rows | Most frequent recorded category | Share among recorded values |
| --- | ---: | ---: | --- | ---: |
| `race` | 1,394 | 2.31% | `Caucasian` | 76.63% |
| `payer_code` | 23,569 | 39.10% | `MC` | 52.44% |
| `medical_specialty` | 29,553 | 49.03% | `InternalMedicine` | 27.93% |
| `diag_1` | 10 | 0.02% | `428` | 6.93% |
| `diag_2` | 216 | 0.36% | `276` | 6.69% |
| `diag_3` | 848 | 1.41% | `250` | 11.52% |

No missing-at-random assumption is established. A high observed mode share does
not imply the same distribution among unrecorded values. Next compare race-only
and payer-only mode imputation separately against the unchanged native CatBoost
reference; leave diagnosis missing markers explicit in this bounded block.

## Native CatBoost with specialty mode imputation — 2026-11-12

Learner executed the same five training-only patient folds with unchanged native
CatBoost settings, imputing only medical_specialty "?" from fold-training modes.
Numerical values remain unscaled; other categorical values are retained.

| Fold | Train AP | Held-out AP | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.249342 | 0.204495 | 0.649960 | 0.094758 |
| 2 | 0.248154 | 0.210334 | 0.657211 | 0.097135 |
| 3 | 0.245334 | 0.209022 | 0.643616 | 0.097707 |
| 4 | 0.242890 | 0.224907 | 0.659652 | 0.099854 |
| 5 | 0.245768 | 0.212160 | 0.651414 | 0.099993 |

Unweighted means from rounded scores: AP ~0.212184, AUROC ~0.652371 and
Brier ~0.097889. Learner-reported mean paired imputed-minus-reference differences
are +0.000896, +0.000339 and -0.000047, respectively. AP and Brier improve in four
folds; AUROC improves in two. These are small descriptive differences, with no
confidence interval or established superiority/calibration conclusion. Full
precision scores/metadata export remains pending. No independent reviewer refit,
saved-validation or final-test scoring in this step. Broader missingness work
remains open; defaults have not been changed based on this result.

Preprocessing policy update (2026-11-09, D-051): RF and LightGBM now bypass
numerical scaling; logistic still scales and CatBoost remains unscaled. The
RF/LightGBM results below used the earlier shared scaled preprocessor and remain
historical snapshots, including tuning/imputation and saved-validation scores.
The regenerated unscaled training-CV comparison is recorded below. No existing
artifact or historical metric has been relabeled as unscaled.

## Unscaled tree references verified — 2026-11-12

The learner reran the same eight LightGBM settings without numerical scaling.
Reviewer checks confirm 40 fold rows, complete fold coverage, summary consistency,
current hashes and scale_numeric=False. Configuration 4 remains first; its means
equal the prior scaled run at saved precision. Some other settings differ: mean
AP for configuration 6 changes by -0.000573042 and for 5 by +0.000430932.
This illustrates why numerical equality was not assumed.

At the learner's explicit request, the reviewer executed fixed Random Forest
on the saved training partition only, with the same five patient-grouped folds,
45 predictors, no numerical scaling, 200 trees, min_samples_leaf=20, seed 42 and
eight threads. Results were exported/read back under
artifacts/random_forest_reference_2026-11-12_unscaled, with automated metadata,
environment/full model parameters and source/split/code/output fingerprints.
Input/code hashes matched before/after fitting. CV took 23.3 seconds in this
environment; no cross-environment timing claim is made.

| Fold | RF train AP | RF held-out AP | RF AUROC | RF Brier |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.282748992 | 0.196862148 | 0.646001458 | 0.096058099 |
| 2 | 0.275205901 | 0.195762989 | 0.652318069 | 0.098616391 |
| 3 | 0.277317097 | 0.199119025 | 0.641237271 | 0.099008665 |
| 4 | 0.275193852 | 0.224275660 | 0.665532145 | 0.101355708 |
| 5 | 0.281124608 | 0.206824832 | 0.644587420 | 0.101355245 |

| Procedure, numerical scaling disabled | Mean AP | Mean AUROC | Mean Brier |
| --- | ---: | ---: | ---: |
| RF fixed reference | 0.204569 | 0.649935 | 0.099279 |
| LightGBM selected configuration 4 | 0.209942 | 0.649015 | 0.097955 |
| Initial native-category CatBoost reference | 0.211288 | 0.652031 | 0.097937 |

All are unweighted fold means, not pooled metrics. CatBoost is the prior
learner-executed reference, not independently rerun here. Tuning budgets and
categorical representations differ; no universal superiority or calibration
conclusion follows. The LightGBM selected score is subject to selection optimism.
No saved-validation/test predictions were generated. Existing synthetic suite
was not rerun because no implementation changed (last full run: 99 passing).

## Initial native-category CatBoost — 2026-11-09

Learner-executed training-only five-fold patient-grouped CV, 45 predictors,
200 iterations, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, seed 42 and eight
CPU threads; no external imputation or one-hot encoding. These are initial
settings, not a matched tuning budget against the selected LightGBM. No
independent reviewer refit. Rounded learner-reported scores:

| Fold | Train AP | Held-out AP | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.247326 | 0.203218 | 0.651195 | 0.094813 |
| 2 | 0.248214 | 0.209739 | 0.657836 | 0.097156 |
| 3 | 0.244296 | 0.207488 | 0.642094 | 0.097760 |
| 4 | 0.243399 | 0.225405 | 0.660616 | 0.099826 |
| 5 | 0.244326 | 0.210590 | 0.648415 | 0.100128 |
| Mean | 0.245512 | 0.211288 | 0.652031 | 0.097937 |

The training-prior dummy scores match the earlier fold baselines. CatBoost's
observed mean AP is slightly above tuned LightGBM configuration 4 (0.209942),
but superiority is not established and no difference CI is computed. Lower
mean Brier alone is not proof of better calibration. The reported 43.0 seconds
include interactive timing and are not a controlled runtime benchmark.
Full-precision artifact/metadata export remains pending; missing-value experiments
remain open under D-050. No saved-validation or final-test scoring in this step.

## Bounded LightGBM tuning — 2026-11-08

Learner reports 40 fold-result rows for eight configurations evaluated on the
same five saved-training patient folds, with 45 predictors and default explicit
missing categories. Trees=200 and learning_rate=0.05 stay fixed. The following
rounded summaries were supplied by the learner, not independently refitted.

| ID | Leaves | Minimum child samples | L2 | Mean AP | Fold AP SD | Mean train AP | Mean AUROC | Mean Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15 | 100 | 10.0 | 0.209942 | 0.011075 | 0.258635 | 0.649015 | 0.097955 |
| 6 | 31 | 20 | 10.0 | 0.209897 | 0.011348 | 0.310425 | 0.649216 | 0.097971 |
| 8 | 31 | 100 | 10.0 | 0.209549 | 0.011651 | 0.305437 | 0.648995 | 0.097989 |
| 2 | 15 | 20 | 10.0 | 0.209229 | 0.010531 | 0.261164 | 0.648565 | 0.097985 |
| 3 | 15 | 100 | 0.0 | 0.209174 | 0.010892 | 0.270961 | 0.647748 | 0.097999 |
| 7 | 31 | 100 | 0.0 | 0.208066 | 0.010546 | 0.333415 | 0.647486 | 0.098090 |
| 1 | 15 | 20 | 0.0 | 0.207925 | 0.010274 | 0.298355 | 0.646282 | 0.098061 |
| 5 | 31 | 20 | 0.0 | 0.206695 | 0.010567 | 0.392332 | 0.646018 | 0.098162 |

Configuration 4 is the provisional candidate under the prespecified mean-AP
selection criterion. Its difference from configuration 6 is only 0.000045 at
reported precision; superiority is not established. Selected CV performance is
subject to selection optimism, and fold SD is not a confidence interval. The
smaller training/CV AP gap is consistent with stronger regularization, not proof
that generalization problems are resolved. Full-precision exports and execution
metadata remain to be saved/verified; this table alone is not a reproducible
experiment artifact. No new saved-validation or final-test scoring in this review.

Artifact update (2026-11-09): the learner saved both full-precision CSVs and
metadata.json under artifacts/lightgbm_tuning_2026-11-08 (ignored by Git).
Read-only checks verify summary consistency with all 40 fold rows, eight
configurations, five current file hashes, feature lists and selection/CV policy.
Environment/provenance metadata explicitly records retrospective capture, not
automatic logging at fit time. These checks do not reproduce training; a
persistent tuning command remains the next implementation step.

Command completion (2026-11-09): the learner implemented and executed
`python -m scripts.tune_lightgbm`. New artifacts under
artifacts/lightgbm_tuning_2026-11-09 passed dimension/fold coverage, aggregation,
configuration and current-file hash checks. Both CSVs are byte-identical to the
earlier interactive exports on this environment. This is a learner-executed
repeat with reviewer artifact inspection, not an independent reviewer refit or
proof of cross-platform determinism. The command refuses an existing output
directory and currently configures that destination in main. Mocked orchestration
tests verify training-only search and CSV-before-metadata ordering; the full
97-test suite passes. Final test predictions remain unevaluated.

## Targeted imputation sensitivity experiment — 2026-11-08

Learner reports five-fold training-CV tables for fixed LightGBM with 45 predictors,
comparing the explicit missing category with the planned medical_specialty-only
most-frequent imputation (D-048). No independent refit or inspection of actual
experiment calls in this review. Means below are unweighted fold summaries
computed from the pasted six-decimal scores, not pooled metrics.

| Metric | Explicit category | Mode imputation | Imputed minus reference |
| --- | ---: | ---: | ---: |
| AP | 0.2079252 | 0.2087450 | +0.0008198 |
| AUROC | 0.6462816 | 0.6473352 | +0.0010536 |
| Brier | 0.0980606 | 0.0980168 | -0.0000438 |

Paired AP differences by fold: -0.002263, +0.003531, +0.001107, +0.002126,
-0.000402. AP improves in three folds and worsens in two. Differences are small
descriptively; these summaries establish neither superiority nor equivalence.
No confidence interval was computed for this experiment. The default explicit
category remains the reference, with optional imputation available. No final-test
or additional saved-validation scoring was performed in this review. Recover
full-precision summaries from the original result DataFrames for final exports.

This is an intermediate Phase 2 report preserving the initial logistic/dummy
reference and subsequent RF/LightGBM CV comparisons. Phase 2 is not complete. AP is reported
explicitly below. Subsequent learner executions produced logistic validation
PR coordinates and trapezoidal area 0.192736, versus AP 0.193178; the area uses
a different interpolation convention. No PR figure has been generated.

## Training-only grouped cross-validation update — 2026-11-05

The learner executed five-fold GroupKFold (shuffle=True, random_state=42) on
the saved training partition only. Each fold fits a fresh complete logistic
pipeline and a separate training-prior dummy. Both score the same held-out
encounters. The six teaching features and outer partitions remain unchanged.
Values below are learner-reported, rounded to six decimals; this update did
not independently rerun real-data CV. Synthetic regression tests verify
fold isolation, learned preprocessing, positive-class scores and metric wiring.

| Fold | AP | Dummy AP | AUROC | Dummy AUROC | Brier | Dummy Brier |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.185501 | 0.110525 | 0.627811 | 0.500000 | 0.095952 | 0.098339 |
| 2 | 0.199489 | 0.113866 | 0.631089 | 0.500000 | 0.097945 | 0.100902 |
| 3 | 0.195535 | 0.114286 | 0.626964 | 0.500000 | 0.098610 | 0.101225 |
| 4 | 0.214590 | 0.118153 | 0.638187 | 0.500000 | 0.100587 | 0.104210 |
| 5 | 0.195646 | 0.117526 | 0.625379 | 0.500000 | 0.101156 | 0.103725 |

Mean fold AP is 0.198152; sample SD across folds (ddof=1) is 0.010550.
These describe equally weighted fold scores, not a confidence interval or
pooled out-of-fold AP; fold fits share training records. Logistic AP/AUROC
exceed dummy and Brier is lower in all five folds. This is descriptive internal
evidence, not proof of better calibration, external validity or clinical utility.
No test predictions have been generated. Broader justified features and common-fold
Random Forest/LightGBM comparisons remain required before Phase 2 closure.

## First feature extension — training-only CV, 2026-11-05

The learner reran the same grouped folds with extended_counts: the initial six
features plus num_lab_procedures, num_procedures, number_outpatient and
number_diagnoses. Logistic settings and preprocessing methods are unchanged.
No real-data rerun was performed by the assistant in this review.

The label extended_counts identifies this historical ten-predictor experiment;
it is no longer a runnable configuration name. Code now retains only initial
and extended. The next historical extension additionally included
admission_source_id (eleven predictors); its results appear in the following
update. The table below refers only to the earlier ten-feature experiment.

| Fold | Extended AP | Extended AUROC | Extended Brier |
| --- | --- | --- | --- |
| 1 | 0.185642 | 0.629200 | 0.095914 |
| 2 | 0.198931 | 0.634881 | 0.097900 |
| 3 | 0.195167 | 0.625738 | 0.098571 |
| 4 | 0.214751 | 0.640214 | 0.100519 |
| 5 | 0.194894 | 0.622807 | 0.101149 |

Dummy metrics are unchanged from the preceding table. Calculated from the rounded
reported fold scores, equal-fold mean differences (extended minus initial) are
AP -0.0002752, AUROC +0.0006820 and Brier -0.0000394. The Brier decrease occurs
in all five folds; AP/AUROC changes vary in direction. These small descriptive
changes do not establish practical relevance, statistical superiority or
equivalence, nor individual feature usefulness. This is an initial four-count
extension; review of the remaining candidate fields and RF/LightGBM comparison
remain pending. The validation snapshot below retains its original six features.

## Admission-source extension — training-only CV, 2026-11-05

Learner-reported execution of extended with eleven predictors: the eight numeric
counts from the preceding experiment plus age, admission_type_id and
admission_source_id. Retain distinct source codes, including unavailable labels;
one-hot encoding is fitted inside each fold. Fold membership and logistic
settings remain unchanged. No independent real-data rerun in this review.

| Fold | AP | AUROC | Brier |
| --- | --- | --- | --- |
| 1 | 0.183745 | 0.626668 | 0.096002 |
| 2 | 0.199190 | 0.635974 | 0.097879 |
| 3 | 0.195684 | 0.626217 | 0.098545 |
| 4 | 0.214796 | 0.640900 | 0.100519 |
| 5 | 0.195042 | 0.623335 | 0.101135 |

Dummy metrics remain unchanged. Calculations using rounded reported values give
mean AP 0.1976914, AUROC 0.6306188 and Brier 0.0988160. Mean paired differences
versus the ten-feature experiment are AP -0.0001856, AUROC +0.0000508 and Brier
+0.0000054. These small descriptive mixed changes do not establish benefit,
equivalence or lack of usefulness of admission_source_id. No clinical claim or
test-based selection is made. Subsequent extended changes must record their
exact predictors separately; these results refer to the eleven listed above.

## Broader configuration — training-only CV, 2026-11-05

The runnable extended configuration now has 45 predictors: all source fields
except encounter_id, patient_nbr, readmitted, weight and discharge_disposition_id.
Eight are numeric counts and 37 are categorical, including separate diagnosis
positions, the 23 drug-specific columns and the two medication indicators.
Preserve recorded missing-information categories (?, literal None and named
unavailable labels). Weight is omitted for this first broader model due to
96.83% training missingness; disposition is omitted as a predictor under the
documented timing ambiguity. Source/cohort records are not deleted by these
feature choices. Diagnosis inclusion assumes availability by discharge without
field timestamps to verify it; absence of temporal leakage is not established.

Extended preprocessing uses sparse one-hot/combined output for these sparse
data; numeric scaling remains fold-fitted on its separate branch. Initial keeps
its original six inputs and dense preprocessing. All 73 synthetic tests pass,
including dense/sparse numerical equivalence on small fixtures. The learner
executed this 45-feature grouped CV and reported no warnings; the assistant did
not independently rerun it. Preceding performance tables remain historical
smaller-feature experiments and must not be relabeled as this model.

| Fold | AP | AUROC | Brier |
| --- | --- | --- | --- |
| 1 | 0.185181 | 0.632036 | 0.096317 |
| 2 | 0.192494 | 0.629207 | 0.098813 |
| 3 | 0.198324 | 0.630323 | 0.098736 |
| 4 | 0.205007 | 0.642185 | 0.101282 |
| 5 | 0.198352 | 0.628234 | 0.101394 |

Dummy metrics are unchanged from the earlier paired folds. From rounded reported
values, equal-fold means are AP 0.1958716, AUROC 0.6323970 and Brier 0.0993084.
Compared with the initial six-feature logistic model, mean paired differences
are AP -0.0022806, AUROC +0.0025110, Brier +0.0004584. The broader model does not
uniformly improve the metrics; lower Brier is preferable. It still exceeds the
dummy in AP/AUROC and has lower Brier in all five folds. These are descriptive
internal comparisons, without a new confidence interval, proof of equivalence,
clinical utility or verified real-time feature availability. Absence of reported
optimizer warnings does not establish a globally optimal predictive configuration.
The broader set is now the common predictor set for the planned model-family
comparison; the simpler initial reference is retained and not declared obsolete.

## First Random Forest comparison — training-only CV, 2026-11-05

Learner-reported execution on the same 45 predictors and grouped folds: 200 trees,
min_samples_leaf=20, bootstrap=True, max_features="sqrt", class_weight=None,
random_state=42, n_jobs=8. Shared fold-fitted preprocessing is unchanged. The
learner measured 21.8 seconds for the CV evaluation after loading the verified
partitions, on their M3 Max with 36 GB. This is one elapsed-time observation,
not a parallelism benchmark. The assistant did not independently rerun RF CV.

| Fold | AP | AUROC | Brier |
| --- | --- | --- | --- |
| 1 | 0.196862 | 0.646012 | 0.096058 |
| 2 | 0.195759 | 0.652317 | 0.098616 |
| 3 | 0.199109 | 0.641233 | 0.099009 |
| 4 | 0.224276 | 0.665532 | 0.101356 |
| 5 | 0.206826 | 0.644589 | 0.101355 |

Dummy scores remain unchanged. From rounded reported fold values, equal-fold
means are AP 0.2045664, AUROC 0.6499366 and Brier 0.0992788. Paired mean RF-minus-
45-feature-logistic differences are AP +0.0086948, AUROC +0.0175396, Brier
-0.0000296. RF AP/AUROC are higher in all five folds; Brier differences vary in
direction and are small on average. No superiority interval has been calculated.
These results do not demonstrate improved calibration or clinical utility; the
initial six-feature model remains a valid simpler reference.

Before choosing an imputation experiment, an assistant read-only input check
found zero missing numeric values in all eight selected counts on saved training.
Categorical missing-information markers are explicitly encoded already; no
specific imputation benefit follows from these performance values. Next is a
training-versus-held-out AP diagnostic, not unstructured tuning to maximize CV.
LightGBM and the final comparative uncertainty/report remain pending.

### In-sample versus held-out RF AP diagnostic

The learner reran the same RF configuration after adding train_ap to the generic
evaluator. Held-out metrics remained unchanged; elapsed time was 22.8 seconds,
now including training predictions, and is not directly a scaling benchmark
against the earlier 21.8-second workload. Reported training AP by fold is
[0.282747, 0.275206, 0.277313, 0.275195, 0.281123]. From rounded values,
mean train AP is 0.2783168 versus held-out AP 0.2045664, a mean gap of
0.0737504. All five gaps are positive (range 0.050919–0.085885).

This is evidence of training optimism consistent with overfitting, not a direct
estimate of recoverable performance or proof that capacity is the sole problem.
Training AP is not a generalization score; fold prevalences also differ slightly.
The extended logistic model needs the same diagnostic for a meaningful comparison
of model-specific gaps. Synthetic tests verify training AP uses fit rows while
the reported AP/AUROC/Brier comparison remains on held-out rows.

The learner subsequently supplied 45-feature logistic training AP by fold:
[0.263136, 0.263307, 0.262546, 0.260921, 0.260733]. Its mean is 0.2621286,
versus held-out mean 0.1958716, giving a mean gap of 0.0662570 (rounded inputs).
Both models have positive in-sample/held-out gaps. The RF gap is somewhat larger
(0.0737504), but this alone neither attributes the gap to a single mechanism nor
establishes that more regularization will improve held-out AP. RF still has
better held-out AP than the broad logistic model. The next model-family comparison
is LightGBM; no imputation/tuning change is justified solely by a gap reduction.

## First LightGBM comparison — training-only CV, 2026-11-05

Learner-reported execution uses LightGBM 4.7.0 on the same 45 predictors and
patient-grouped folds. Configuration: binary objective, 200 boosting rounds,
learning_rate=0.05, num_leaves=15, min_child_samples=20, random_state=42,
n_jobs=8. Shared fold-fitted sparse preprocessing remains unchanged. Measured
elapsed time is 22.1 seconds including training/held-out predictions; no
independent real-data rerun by the assistant in this review.

| Fold | Training AP | Held-out AP | AUROC | Brier |
| --- | --- | --- | --- | --- |
| 1 | 0.299535 | 0.198108 | 0.645092 | 0.095041 |
| 2 | 0.295958 | 0.205093 | 0.647102 | 0.097424 |
| 3 | 0.300875 | 0.199846 | 0.637526 | 0.098038 |
| 4 | 0.296219 | 0.222631 | 0.655752 | 0.099860 |
| 5 | 0.299190 | 0.213948 | 0.645936 | 0.099940 |

Dummy scores are unchanged. Equal-fold means from rounded reported inputs:

| Model, 45 inputs | Training AP | Held-out AP | AUROC | Brier |
| --- | --- | --- | --- | --- |
| Logistic | 0.262129 | 0.195872 | 0.632397 | 0.099308 |
| Random Forest | 0.278317 | 0.204566 | 0.649937 | 0.099279 |
| LightGBM | 0.298355 | 0.207925 | 0.646282 | 0.098061 |

LightGBM has the highest mean held-out AP and lowest Brier, while RF has the
highest mean AUROC. LightGBM-minus-RF mean paired differences are AP +0.0033588,
AUROC -0.0036550 and Brier -0.0012182. LightGBM Brier is lower in all five folds;
AP is higher in four. Its mean AP training gap is also largest (0.0904302).
No single model dominates every metric, and no uncertainty interval has yet
been computed for these model-family differences. These are descriptive CV
results, not a clinical-use or calibration claim.

LightGBM logs are informational. Reported used-feature counts 805/803/789/797/795
refer to internal encoded features, not source columns. One-hot expansion and
LightGBM's default prefilter for unsplittable features can explain why this count
differs from both 45 raw fields and the full encoder width; this review did not
trace which individual features were filtered. Fold vocabularies/support differ.
The initial score uses fold-training prevalence transformed to log odds. Automatic
row-wise selection and its roughly 0.004-second timing do not require a parameter
change. Logs do not prove predictive usefulness of every retained feature.

Next: hold the three initial configurations fixed, fit fresh pipelines on the
entire saved training partition and evaluate on the same saved outer validation
for calibration and paired patient-cluster uncertainty. Outer validation is
development data already inspected for the initial six-feature baseline, not
an untouched final test. The final test remains reserved.

## Fixed configurations on saved validation — 2026-11-06

Learner-reported execution refits each of the three fixed 45-feature pipelines
on all 60,276 saved training encounters and evaluates the same 19,883 saved
validation encounters (2,233 positives). Preprocessing is fitted only on training.
This holdout is development data already inspected for the initial baseline,
not external-source validation or the untouched final test. The assistant did
not independently rerun these fits.

| Model | Encounters | AP | AUROC | Brier |
| --- | --- | --- | --- | --- |
| Logistic | 19,883 | 0.193949 | 0.636235 | 0.097040 |
| Random Forest | 19,883 | 0.205392 | 0.652106 | 0.097238 |
| LightGBM | 19,883 | 0.202084 | 0.647656 | 0.096249 |

RF has the highest point AP/AUROC, LightGBM the lowest Brier. RF-minus-LightGBM
AP is +0.003308; LightGBM-minus-RF Brier is -0.000989. Calibration and uncertainty
for these differences are not established by this table. The AP ordering differs
from equal-fold mean CV ordering; different evaluation patients and full-training
fits make this possible without demonstrating a split error. Do not replace these
values with a preferred fold/seed result or claim an unqualified winner.

LightGBM logs report 6,925 positives/53,351 negatives and 60,276 training rows,
consistent with saved training. Its 870 used encoded features need not match
fold-training counts because the available categories and split support differ.
No prediction-time or clinical-utility guarantee follows from these checks.
Next: comparative fixed-bin calibration summaries with encounter/patient support,
then paired patient-cluster bootstrap using aligned predictions on this holdout.

## Saved-validation calibration summaries — 2026-11-06

Learner-reported execution of scripts.evaluate_models reproduces the preceding
holdout metrics and prints the same predefined probability bins for each model.
No calibration mapping is fitted and no probability is changed. First bin includes
zero; the pandas display (-0.001, 0.1] is an inclusion convention, not negative
predicted probabilities. Empty bins are omitted by observed=True.

| Model | Bin | Encounters | Patients | Positives | Mean prediction | Observed fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic | [0, 0.1] | 10,510 | 9,238 | 802 | 0.066790 | 0.076308 |
| Logistic | (0.1, 0.2] | 7,558 | 5,818 | 993 | 0.135784 | 0.131384 |
| Logistic | (0.2, 0.3] | 1,301 | 1,040 | 266 | 0.236947 | 0.204458 |
| Logistic | (0.3, 0.4] | 342 | 272 | 103 | 0.340144 | 0.301170 |
| Logistic | (0.4, 0.5] | 99 | 72 | 35 | 0.441920 | 0.353535 |
| Logistic | (0.5, 1] | 73 | 37 | 34 | 0.620200 | 0.465753 |
| RF | [0, 0.1] | 6,500 | 6,083 | 393 | 0.086902 | 0.060462 |
| RF | (0.1, 0.2] | 13,234 | 8,986 | 1,781 | 0.126697 | 0.134578 |
| RF | (0.2, 0.3] | 149 | 74 | 59 | 0.213696 | 0.395973 |
| LightGBM | [0, 0.1] | 10,094 | 9,459 | 734 | 0.074754 | 0.072716 |
| LightGBM | (0.1, 0.2] | 8,467 | 6,099 | 1,137 | 0.134738 | 0.134286 |
| LightGBM | (0.2, 0.3] | 958 | 688 | 229 | 0.232239 | 0.239040 |
| LightGBM | (0.3, 0.4] | 253 | 148 | 86 | 0.345017 | 0.339921 |
| LightGBM | (0.4, 0.5] | 90 | 41 | 37 | 0.437228 | 0.411111 |
| LightGBM | (0.5, 1] | 21 | 15 | 10 | 0.547398 | 0.476190 |

Each model's encounter/positive totals match 19,883/2,233. Patient counts are
not additive across bins, as one patient's encounters may occupy different bins.
Bin membership differs across models; these rows do not constitute matched
within-bin comparisons of the same patients. Summaries are encounter-weighted,
not patient-weighted, and have no bin-level uncertainty intervals yet.

Logistic underpredicts in the lowest bin and overpredicts in higher bins in this
sample. RF overpredicts in the lowest bin and underpredicts in the upper observed
bin (149 encounters/74 patients). LightGBM's means are close to observed fractions
in its largest bins; its top bin has only 21 encounters/15 patients. This is
descriptive evidence compatible with better calibration, not a verified global
calibration ranking or evidence of clinical readiness. Do not recalibrate and
evaluate on these same outcomes. Next: paired patient-cluster uncertainty for
Brier differences across the fixed models using identical resampled positions.

Bootstrap integration update (2026-11-07): learner execution returns a 1,000-by-3
Brier score table for logistic/RF/LightGBM using shared patient-cluster draws
(default seed 123). The displayed first five rows are a sanity check, not the
distribution used alone for intervals. Paired difference quantiles are not yet
reported. The assistant reviewed the invocation and its alignment test without
independently rerunning source-data models/bootstrap.

First paired interval (learner-reported, 2026-11-07): original saved-validation
Brier difference LightGBM minus RF = -0.000989; 95% patient-cluster bootstrap
percentile CI [-0.001567, -0.000487], using the full 1,000 paired replicates.
Both endpoints are negative, supporting lower Brier for LightGBM under the
stated conditional sampling assumptions. This is not proof of better calibration
alone, clinical utility or variability across new training fits. Percentiles
refer to replicate Brier differences, not patient predicted probabilities; no
patients/predictions are removed by calculating the 2.5%/97.5% quantiles.
The assistant verified the calculation/wiring in code and a synthetic test,
without independently rerunning the source-data interval. Other paired model
comparisons remain pending; intervals will be individual exploratory 95%
intervals, not simultaneous family-wise coverage guarantees.

Remaining paired Brier intervals (learner-reported, 2026-11-07), using the same
replicate table and subtraction order shown:

| Difference | Point estimate | Individual 95% percentile CI |
| --- | --- | --- |
| LightGBM - RF | -0.000989 | [-0.001567, -0.000487] |
| LightGBM - logistic | -0.000791 | [-0.001196, -0.000353] |
| RF - logistic | +0.000198 | [-0.000436, +0.000904] |

The first two intervals support lower Brier for LightGBM under the conditional
patient-cluster sampling assumptions. RF-versus-logistic includes zero and both
signs: this procedure does not establish a clear direction of the Brier difference,
nor does it establish equivalence. Inclusion of zero does not mean that 95% of
bootstrap differences equal zero; the interval endpoints are quantiles of the
replicate difference distribution. No inference about zero-valued replica counts
can be made from these two endpoints alone. The assistant reviewed the loop and
extended its synthetic test to check signs/quantiles for all three pair labels;
no independent source-data interval rerun was performed.

## Paired ranking uncertainty — RF versus LightGBM, 2026-11-07

Learner-reported execution creates a separate 1,000-by-3 AP bootstrap table with
average_precision_score, preserving the Brier table and patient-cluster scheme.
Original validation AP difference RF minus LightGBM is +0.003308; its individual
95% percentile interval is [-0.002945, +0.010011]. The interval includes zero,
so this procedure does not establish a clear direction of AP superiority.
This does not establish equivalence, nor imply that any given proportion of
replicates equals zero. Higher AP is preferable, unlike Brier loss.

The combination of lower Brier intervals for LightGBM and unresolved AP
superiority of RF can motivate LightGBM as a probability-loss-oriented candidate,
without claiming universal superiority or clinical utility. RF still has higher
point AP/AUROC; no AUROC difference interval has been calculated. Patient-level
calibration uncertainty and variability from training/model selection are not
covered by these fixed-model score intervals. All comparisons remain exploratory
development analyses with individual, unadjusted intervals and an untouched
final test. The assistant reviewed the AP call/point/quantile wiring and a
synthetic orchestration test without independently rerunning the source-data fit
or bootstrap.

## Study question and cohort

Estimate the probability of recorded readmission category `<30` at hospital
discharge. Each row is an encounter; repeated encounters from a patient are
retained. This report evaluates generalization to patients absent from training.

The source is the public [UCI Diabetes 130-US Hospitals dataset, 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
The local source contains 101,766 encounters and 71,518 patients. Excluding
discharge disposition codes `11`, `19`, `20`, `21` retains 100,114 encounters
from 70,439 patients. Hospice and transfers remain included. Source data are
preserved; exclusions are applied to a derived cohort.

The binary outcome is `1` for `<30`, and `0` for `>30` or `NO`. `NO` denotes no
recorded readmission, without establishing complete follow-up. See
[COHORT_POLICY.md](COHORT_POLICY.md) for the full contract.

## Patient-disjoint partitions

| Partition | Encounters | Patients | Positive encounters | Positive fraction |
| --- | ---: | ---: | ---: | ---: |
| Training | 60,276 | 42,263 | 6,925 | 11.49% |
| Validation | 19,883 | 14,088 | 2,233 | 11.23% |
| Test | 19,955 | 14,088 | 2,199 | 11.02% |

GroupShuffleSplit first reserves 20% of patients for test, then 25% of development
patients for validation; both calls use seed `42`. Membership is saved and
validated by encounter/patient identifiers. Test support above comes from the
split audit; **no model predictions or performance metrics on test have been
generated**. All results below concern validation encounters, with equal weight
per encounter.

## Inputs and fitting

- Numeric: `time_in_hospital`, `num_medications`, `number_emergency`, `number_inpatient`.
- Categorical: `age`, `admission_type_id`.
- Identifiers and `readmitted` are excluded from predictors. This small initial
  teaching subset is not an optimized feature list or evidence against other fields.
- Numeric strings are converted with errors raised on invalid values. This
  subset has no imputation step. StandardScaler and OneHotEncoder
  (`handle_unknown="ignore"`, dense output) are fitted on training only.
- Age bins receive one-hot encoding, without imposing a linear age effect.
- The pipeline classifier is LogisticRegression with `C=1.0`, `solver="lbfgs"`,
  `max_iter=1000`, default L2 regularization and no class weighting. No tuning
  was performed. The real fit completed in 31 iterations without a convergence warning.
- The reference is DummyClassifier (`strategy="prior"`), fitted on training
  labels. It assigns constant positive probability `6,925 / 60,276`, approximately
  `0.114888`, ignoring predictor values.

Discharge-time feature availability is assumed for this retrospective exercise;
field recording timestamps are unavailable. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Validation probability metrics

| Metric | Logistic | Prior dummy | Preferred direction |
| --- | ---: | ---: | --- |
| Binary Brier score | 0.097085 | 0.099701 | Lower |
| AUROC | 0.634911 | 0.500000 | Higher |
| Average Precision (AP) | 0.193178 | 0.112307 | Higher |

Metrics use the positive-class probability vector with no classification
threshold. AP is sklearn's non-interpolated average precision, distinct from
trapezoidal PR-area integration. Constant-score dummy AP equals validation
prevalence, whereas its predicted probability comes from training prevalence.

## Illustrative validation thresholds

Predictions use `p >= threshold`. These two fixed thresholds illustrate behavior;
neither has been selected as optimal or justified for clinical decisions.

| Threshold | TN | FP | FN | TP | Flagged | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 17,610 | 40 | 2,200 | 33 | 73 | 45.21% | 1.48% |
| 0.20 | 16,904 | 746 | 1,947 | 286 | 1,032 | 27.71% | 12.81% |

## Descriptive calibration support

Fixed right-closed probability bins are used, including zero in the first bin.
Only nonempty bins are displayed. Patient counts can overlap across bins.

| Probability bin | Encounters | Patients | Positives | Mean prediction | Observed positive fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[0, 0.1]` | 10,367 | 9,625 | 804 | 0.086260 | 0.077554 |
| `(0.1, 0.2]` | 8,484 | 6,139 | 1,143 | 0.125928 | 0.134724 |
| `(0.2, 0.3]` | 718 | 487 | 179 | 0.236889 | 0.249304 |
| `(0.3, 0.4]` | 163 | 114 | 52 | 0.341724 | 0.319018 |
| `(0.4, 0.5]` | 78 | 49 | 22 | 0.441240 | 0.282051 |
| `(0.5, 1.0]` | 73 | 22 | 33 | 0.633690 | 0.452055 |

These are grouped descriptive estimates; bin widths and support differ. No
calibration-specific confidence intervals or recalibrator have been fitted.

## Conditional uncertainty

For 1,000 bootstrap replicates with NumPy seed `123`, resample 14,088 validation
patients uniformly with replacement, retaining every encounter for each draw,
including repeated draws. Models and predicted probabilities remain fixed.
Intervals use bootstrap quantiles `0.025` and `0.975` (percentile method).

| Quantity | Original estimate | Approximate 95% patient-cluster percentile CI |
| --- | ---: | --- |
| Logistic Brier | 0.097085 | [0.093205, 0.101404] |
| Brier difference: logistic minus dummy | -0.002616 | [-0.003451, -0.001894] |

The comparison is paired: both losses use identical sampled encounters. Negative
differences favor logistic. These intervals address validation-sampling
uncertainty conditional on the trained models; training variability and external
validity are outside their scope. Bootstrap mean is not substituted for the
original point estimate. No exact finite-sample coverage guarantee is claimed.

## Interpretation

On this validation set, the lower logistic Brier score and the entirely negative
paired bootstrap interval support lower probability loss than the prior dummy
for the fixed models under the stated sampling assumptions.

Lowering the threshold from 0.50 to 0.20 increased detected positive encounters
from 33 to 286 and false positives from 40 to 746, illustrating a recall–precision
trade-off and greater review workload.

The bootstrap quantifies validation-sampling uncertainty conditional on the
fitted models, while training variability and clinical usefulness remain
unassessed and require separate evaluation.

## Study limitations

Historical selected data, incomplete readmission ascertainment and unverified
field timing limit interpretation. Death-code exclusions do not prove complete
mortality ascertainment. This initial six-feature model has not undergone broader
feature experiments, grouped cross-validation, subgroup evaluation or external
validation. Upper calibration bins contain few distinct patients. No final test
evaluation or real-world deployment claim is made.

## Reproduction and evidence

Run from the repository root:

```bash
.venv/bin/python -m scripts.train_baseline
.venv/bin/python -m unittest discover -s tests -v
```

The training command verifies source SHA-256 and saved assignment integrity.
Source fingerprint:
`0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`.
Environment verified: Python 3.13.5, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1. The full 66-test suite passed for this snapshot, and the
actual 1,000-replicate command reproduced the reported outputs. Data and split
artifacts remain ignored by Git; no fitted model is persisted and clean-machine
reconstruction has not been demonstrated. Implementation:
[training command](../scripts/train_baseline.py),
[features/preprocessing](../carepath_lab/features.py),
[model](../carepath_lab/modeling.py),
[patient resampling](../carepath_lab/uncertainty.py).
