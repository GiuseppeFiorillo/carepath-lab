# Optional reproducibility studies

These commands are separate from the current `scripts.train_model` /
`scripts.audit_model` workflow. Run from the repository root after source data
and saved patient splits are available. All output directories must be new.

| Module | Purpose | Feature definition |
| --- | --- | --- |
| `experiments.phase2_comparison` | Reproduce the frozen four-model Phase 2 development comparison | Historical `extended`, 45 predictors |
| `experiments.tune_lightgbm` | Repeat the bounded eight-setting, training-only grouped search | Historical `extended`, 45 predictors |
| `experiments.review_features` | Repeat eight declared feature-ablation profiles (40 fits) | 45-field reference and declared exclusions |

```bash
.venv/bin/python -m experiments.phase2_comparison --output-dir artifacts/phase2_repeat --bootstrap 1000
.venv/bin/python -m experiments.tune_lightgbm --output-dir artifacts/lightgbm_repeat
.venv/bin/python -m experiments.review_features --output-dir artifacts/feature_review_repeat
```

No study automatically replaces the current feature definition, selects a new
winner from development validation, or scores the reserved test. The fixed
current 41-feature choice is documented in [FEATURE_REVIEW](../docs/FEATURE_REVIEW.md)
and versioned under `docs/reports/feature_review_selection.json`.

Prior results retain their original feature/parameter meaning. Moving command
modules changes code paths/hashes; new executions produce their own provenance,
not byte-identical copies of old metadata. The former `scripts.train_baseline`
and `scripts.evaluate_models` teaching commands are superseded; their original
source remains in Git history (for example, feature-review commit `c3b8e95`).
