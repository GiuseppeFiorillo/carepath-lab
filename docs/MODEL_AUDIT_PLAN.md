# Phase 3 — Model Audit Plan

Started 2026-11-13. Scope follows the learner's original Explainable AI and
fairness phase. This is a protocol/learning plan, not computed audit findings.

## Fixed model and evaluation boundary

Originally use the training-fitted specialty-mode CatBoost pipeline and saved
predictions from artifacts/phase2_fixed_validation_2026-11-12. On 2026-11-20,
the learner explicitly authorized a [bounded predictor review](FEATURE_REVIEW.md).
The current 41-feature workflow is scripts.train_model -> scripts.audit_model,
using artifacts/current_model by default. The original revision snapshot remains
in artifacts/feature_review_selected_final_2026-11-20; earlier audit results remain
historical and must not be attributed to it. Audit the 19,883
development-validation encounters, not the final test.
This validation is already inspected development data, not external confirmation.
Source/grouping/feature-availability limitations remain in PHASE_2_REPORT.

## 1. Integrity and support before subgroup metrics

- Load encounter_id/patient_nbr as strings. Join saved selected-model scores and
  binary outcomes to validation demographics using encounter_id with one-to-one
  validation. Verify patient pairs and target consistency; do not rely on row
  order or DataFrame index. Check 19,883 retained encounters and finite scores.
- Initially audit age intervals, recorded gender and recorded race separately.
  These are source labels, not independently verified demographic identities.
  Keep "?" and Unknown/Invalid explicit; do not assign them to known groups.
- Report encounters, distinct patients, positive/negative counts and prevalence.
  Inspect support before fixing disclosure/low-support flags. Encounter count
  alone does not establish independent evidence. Avoid multiplying tiny groups
  by starting with all intersections.

## 2. Explainability

Teach local versus aggregate attribution, reference/background assumptions and
raw-score/log-odds versus probability units before choosing an implementation.
For this native-category model, verify CatBoost/TreeSHAP compatibility and pass
the exact training-fitted preprocessing output with original feature names.
CatBoost's native SHAP computation is an option; a plotting library is a separate
choice, not a reason to change the selected model. Check additivity numerically.
Explain fixed validation examples or a declared reproducible sample, not cherry-
picked successes. If background data are required, draw them from training.
Document dependence between predictors and the absence of causal interpretation.

## 3. Subgroups and thresholds

Plan common exploratory thresholds 0.10, 0.20, 0.30 and 0.50, without selecting
a clinical operating point or optimizing group-specific thresholds. Show
confusion counts and denominators with recall TP/(TP+FN), precision TP/(TP+FP)
and FPR FP/(FP+TN). Undefined metrics are NaN/explicitly undefined, not zero.
Fixed-bin calibration uses continuous probabilities and does not itself depend
on a classification threshold; the selected action/error counts do.

Report support and uncertainty before interpreting gaps. Repeated encounters
require patient-cluster uncertainty; if a patient has encounters in multiple
age groups, preserve those encounters together when resampling the full audit
table. Tiny groups or bootstrap replicates with undefined denominators require
transparent handling, not silent selection of favorable replicas.

Differences may reflect label/capture differences, case mix or dataset selection;
they neither prove causal discrimination nor establish fairness. No single
fairness metric proves equity; no legal compliance or clinical utility claim.

## Deliverable and teaching ownership

Build an automated Model Audit Report incrementally: support tables, SHAP
figures, threshold/subgroup metrics, descriptive calibration and limitations.
Learner writes/understands joins, denominators and interpretation. Reviewer may
add tests and export utilities. Start with pandas/sklearn already installed;
introduce SHAP/Fairlearn tooling only after its role and compatibility are clear.
Do not build the whole report in one step or install unrelated infrastructure.

Primary references: [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
and [Fairlearn assessment](https://fairlearn.org/main/user_guide/assessment/index.html).

## Implemented checkpoint — descriptive subgroup audit

As of 2026-11-19, carepath_lab/audit.py validates one-to-one encounter alignment,
patient/outcome agreement and numeric score/outcome domains, and computes
subgroup metrics and fixed-bin calibration. scripts/audit_model.py exports
aggregate tables with input/code/output hashes and execution settings to a new
local artifact directory. It never refits the selected model or scores the test.

The first uncertainty exercise is deliberately limited to recall Female minus
Male at threshold 0.20: 1,000 shared full-validation patient-cluster draws,
seed 123. The observed difference is about -0.55 percentage points. With the
current outer-join row order, the percentile 95% interval is [-5.51, +4.18]
points. Patient ordering affects finite draws; it does not change the observed
statistic. Replicates with absent groups/undefined recall remain NaN, and an
interval is reported only if all differences are finite. This is a conservative
reporting rule, not a remedy for inadequate subgroup support.

These numerical recall results belong to the historical 45-feature model;
they must not be attributed to the current 41-feature revision.

No equivalence margin or TOST conclusion is claimed. Native SHAP checks,
bar/beeswarm/waterfall exports and a three-sample descriptive sensitivity study
are now documented in the [XAI checkpoint](MODEL_AUDIT_REPORT.md). Broader subgroup
uncertainty and the final integrated Model Audit Report remain pending;
these checkpoints do not complete Phase 3.
