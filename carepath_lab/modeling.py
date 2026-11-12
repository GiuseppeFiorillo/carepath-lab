"""Build classifier pipelines and evaluate them with patient-grouped CV."""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from carepath_lab.features import build_model_inputs, build_preprocessor, FEATURE_SETS
from carepath_lab.splitting import make_patient_folds


def build_baseline_model(feature_set: str = "initial") -> Pipeline:
    """Return an unfitted preprocessing and logistic-regression pipeline."""
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_set=feature_set)),
            ("classifier", classifier),
        ]
    )


def evaluate_model_cv(
    train: pd.DataFrame,
    feature_set: str = "initial",
    model_builder=None,
) -> pd.DataFrame:
    """Compare a model builder with a prior dummy on grouped training folds.

    Pass only the saved training partition. The builder must accept feature_set
    and return a fresh unfitted pipeline; omission selects logistic regression.
    Each fold fits preprocessing, classifier and dummy on its training rows,
    then reports AP, AUROC and Brier on that fold's held-out encounters.
    Also report in-sample train_ap as a fit diagnostic, not generalization.
    """
    if model_builder is None:
        model_builder = build_baseline_model

    X, y = build_model_inputs(train, feature_set=feature_set)
    fold_results = []

    for fold, (fit_positions, validation_positions) in enumerate(
        make_patient_folds(train), start=1
    ):
        model = model_builder(feature_set=feature_set)

        model.fit(X.iloc[fit_positions], y.iloc[fit_positions])
        fold_positive_probabilities = model.predict_proba(
            X.iloc[validation_positions]
        )[:, 1]

        fit_positive_probabilities = model.predict_proba(
            X.iloc[fit_positions]
        )[:, 1]

        dummy_model = DummyClassifier(strategy="prior")
        dummy_model.fit(X.iloc[fit_positions], y.iloc[fit_positions])
        dummy_positive_probabilities = dummy_model.predict_proba(
            X.iloc[validation_positions]
        )[:, 1]

        fold_results.append(
            {
                "fold": fold,
                "train_ap": average_precision_score(
                    y.iloc[fit_positions],
                    fit_positive_probabilities,
                ),
                "ap": average_precision_score(
                    y.iloc[validation_positions],
                    fold_positive_probabilities,
                ),
                "dummy_ap": average_precision_score(
                    y.iloc[validation_positions],
                    dummy_positive_probabilities,
                ),
                "auroc": roc_auc_score(
                    y.iloc[validation_positions],
                    fold_positive_probabilities,
                ),
                "dummy_auroc": roc_auc_score(
                    y.iloc[validation_positions],
                    dummy_positive_probabilities,
                ),
                "brier": brier_score_loss(
                    y.iloc[validation_positions],
                    fold_positive_probabilities,
                ),
                "dummy_brier": brier_score_loss(
                    y.iloc[validation_positions],
                    dummy_positive_probabilities,
                ),
            }
        )

    return pd.DataFrame(fold_results)


def build_random_forest_model(
    feature_set: str = "initial", n_jobs: int = 8
) -> Pipeline:
    """Build an unfitted forest pipeline with configurable tree parallelism."""
    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=n_jobs,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(feature_set=feature_set, scale_numeric=False),
            ),
            ("classifier", classifier),
        ]
    )


def build_lightgbm_model(
    feature_set: str = "initial", 
    n_jobs: int = 8,
    imputed_categorical_features: tuple[str, ...] = (),
    num_leaves: int = 15,
    min_child_samples: int = 20,
    reg_lambda: float = 0.0,
) -> Pipeline:
    """Build an unfitted binary boosting pipeline with configurable CPU threads."""
    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=42,
        n_jobs=n_jobs,
        reg_lambda=reg_lambda,
    )

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(
                feature_set=feature_set,
                imputed_categorical_features=imputed_categorical_features,
                scale_numeric=False,
            )),
            ("classifier", classifier),
        ]
    )

def fit_and_predict_validation(
    train,
    validation,
    model_builder,
    feature_set="initial",
):
    """Fit on saved training and predict on the separate saved validation holdout."""
    X_train, y_train = build_model_inputs(train, feature_set=feature_set)
    X_validation, y_validation = build_model_inputs(
        validation, feature_set=feature_set
    )
    model = model_builder(feature_set=feature_set)

    model.fit(X_train, y_train)
    positive_probabilities = model.predict_proba(X_validation)[:, 1]

    return model, y_validation, positive_probabilities


def build_catboost_model(
    feature_set: str = "initial",
    n_jobs: int = 8,
    imputed_categorical_features: tuple[str, ...] = (),
    add_missing_indicators: bool = False,
) -> Pipeline:
    categorical_features = FEATURE_SETS[feature_set]["categorical"]

    classifier = CatBoostClassifier(
        cat_features=categorical_features,
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=42,
        thread_count=n_jobs,
        verbose=False,
        allow_writing_files=False,
    )

    steps = []
    if imputed_categorical_features:
        steps.append(
            (
                "preprocessor",
                build_preprocessor(
                    feature_set=feature_set,
                    imputed_categorical_features=imputed_categorical_features,
                    native_categorical=True,
                    scale_numeric=False,
                    add_missing_indicators=add_missing_indicators,
                )
            )
        )
    
    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)
