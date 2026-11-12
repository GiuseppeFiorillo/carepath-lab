# Literature notes for Phase 2 improvement

Reviewed: 2026-11-07. These sources motivate experiments; they are not reproduced benchmarks. Compare target definition, prevalence, exclusions, patient separation, feature availability and metric definition before comparing scores. AP and trapezoidal PR area must not be silently interchanged.

## Closest identified patient-disjoint reference

Bhuvan Rajpoot, *Does Discharge-Time Information Improve 30-Day Readmission Prediction? A Patient-Disjoint Evaluation in the UCI Diabetes 130-US Hospitals Cohort* (2026), [SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7154739).

The indexed abstract reports patient-disjoint evaluation, development-only grouped validation, 99,340 encounters after exclusions, and discharge-time histogram gradient boosting with AP 0.237 and AUROC 0.677 at prevalence 11.52%. It excludes hospice pathways and invalid gender as well as deaths, unlike our cohort. Its features and development protocol also differ. This is a preprint; peer review was not verified. The full text could not be accessed (403), so implementation and detailed methods remain unverified. This is evidence of a relevant reported result, not a promised target or a directly paired comparison.

## Published comparison of common model families

Emi-Johnson and Nkrumah, *Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data* (2025), [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/), [DOI](https://doi.org/10.7759/cureus.82437).

The study compares logistic regression, Random Forest, XGBoost and a deep neural network. It reports AUROC 0.667 for XGBoost and 0.579 for the neural network. These are AUROC values, not AP. The read methods describe a stratified 80/20 split; patient-disjoint separation was not established from those methods. Its numbers therefore do not define a fair performance target for our grouped protocol. Complexity alone did not guarantee a better result in that comparison.

## Candidate methods and engineering implications

- [CatBoost categorical-feature documentation](https://catboost.ai/docs/en/features/categorical-features): supports categorical inputs and advises against external one-hot preprocessing. It is a plausible challenger for our predominantly categorical predictors, not an established winner on this particular cohort. CatBoost has not been installed or evaluated here.
- [LightGBM categorical-feature support](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support): supports categorical splits, including automatic detection of unordered pandas categorical columns. Our existing pipeline instead uses one-hot encoding. Native categories are a possible separate experiment, not equivalent to passing arbitrary numeric codes as quantitative measurements.

## Planned next step

Preserve all baseline results. Start with the eight-configuration LightGBM training-CV experiment in D-047, keeping representation fixed so the comparison addresses hyperparameters. Then evaluate a CatBoost categorical-input challenger using the same patients/folds. Selection uses AP; retain Brier/AUROC, fold prevalence and configuration details. The final test remains reserved. Neither this literature review nor the current AP establishes a performance ceiling, clinical utility or successful improvement.
