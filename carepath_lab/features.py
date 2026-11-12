"""Declare named feature sets and construct typed model inputs."""

from multiprocessing import Value

from narwhals._compliant import column
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_medications",
    "number_emergency",
    "number_inpatient",
]

CATEGORICAL_FEATURES = [
    "age",
    "admission_type_id",
]

# Drug-specific categorical status; separate from change/diabetesMed indicators.
MEDICATION_FEATURES = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

FEATURE_SETS = {
    "initial": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "extended": {
        "numeric": [
            *NUMERIC_FEATURES,
            "num_lab_procedures",
            "num_procedures",
            "number_outpatient",
            "number_diagnoses",
        ],
        "categorical": [
            *CATEGORICAL_FEATURES,
            "admission_source_id",
            "max_glu_serum",
            "A1Cresult",
            *MEDICATION_FEATURES,
            "change",
            "diabetesMed",
            "race",
            "gender",
            "payer_code",
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
        ],
    },
}


def build_model_inputs(
    data: pd.DataFrame,
    feature_set: str = "initial",
) -> tuple[pd.DataFrame, pd.Series]:
    """Select typed predictors and map valid recorded outcomes to a binary target."""
    feature_config = FEATURE_SETS[feature_set]
    numeric_features = feature_config["numeric"]
    categorical_features = feature_config["categorical"]

    allowed_readmitted_values = {"<30", ">30", "NO"}
    observed_readmitted_values = set(data["readmitted"].unique())
    unexpected_readmitted_values = observed_readmitted_values - allowed_readmitted_values

    if unexpected_readmitted_values:
        raise ValueError(
            f"Unexpected readmitted categories: {sorted(unexpected_readmitted_values)}."
        )

    readmitted_map = {"<30": 1, ">30": 0, "NO": 0}
    X = data[numeric_features + categorical_features].copy()
    y = data["readmitted"].map(readmitted_map)

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="raise")

    return X, y


def attach_split_assignments(
    cohort: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Attach partition labels to a cohort using validated encounter assignments."""
    merged_df = cohort.merge(
        assignments[["encounter_id", "partition"]],
        on="encounter_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    return merged_df


def build_preprocessor(
    feature_set: str = "initial",
    imputed_categorical_features: tuple[str, ...] = (),
    native_categorical: bool = False,
    scale_numeric: bool = True,
    add_missing_indicators: bool = False,
) -> ColumnTransformer:
    """Build an unfitted preprocessor for numeric and categorical predictors."""
    feature_config = FEATURE_SETS[feature_set]
    numeric_features = feature_config["numeric"]
    categorical_features = feature_config["categorical"]

    invalid_imputation_features = set(imputed_categorical_features) - set(categorical_features)
    if invalid_imputation_features:
        raise ValueError(f"Invalid imputation features: {sorted(invalid_imputation_features)}")

    if len(imputed_categorical_features) != len(set(imputed_categorical_features)):
        raise ValueError("Duplicate imputation features are not allowed.")

    non_imputed_categorical_features = [
        column
        for column in categorical_features
        if column not in imputed_categorical_features
    ]

    use_sparse = feature_set == "extended"

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    categorical_transformer = (
        "passthrough"
        if native_categorical
        else OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=use_sparse,
        )
    )
    
    imputation_steps = [
        (
            "imputer",
            SimpleImputer(
                missing_values="?",
                strategy="most_frequent",
                add_indicator=add_missing_indicators,
            )
        )
    ]

    if not native_categorical:
        imputation_steps.append(
            (
                "encoder", 
                OneHotEncoder(
                    handle_unknown="ignore", 
                    sparse_output=use_sparse
                )
            )
        )

    categorical_imputation_pipeline = Pipeline(steps=imputation_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            (
                "categorical",
                categorical_transformer,
                non_imputed_categorical_features,
            ),
            (
                "categorical_imputed",
                categorical_imputation_pipeline,
                list(imputed_categorical_features)
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if use_sparse else 0.3,
        verbose_feature_names_out=not native_categorical,
    )

    if native_categorical:
        preprocessor.set_output(transform="pandas")

    return preprocessor
