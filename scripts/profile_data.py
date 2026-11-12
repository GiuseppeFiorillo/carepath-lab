"""Profile source values, categories, duplicates and the declared analysis cohort."""

from pathlib import Path

import pandas as pd


from carepath_lab.cohort import select_analysis_cohort


def print_section(title):
    """Separate console sections without changing exported data."""
    print(f"\n{title}\n{'-' * len(title)}")


def audit_numeric_counts(values, label):
    """Print one row of the numeric audit table."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    non_integer_values = (numeric_values.dropna() % 1) != 0
    negative_values = numeric_values < 0
    print(
        f"{label:<26} | {numeric_values.min():>7g} | {numeric_values.max():>7g} | "
        f"{numeric_values.isna().sum():>11} | {non_integer_values.sum():>10} | "
        f"{negative_values.sum():>8}"
    )


def print_cohort_summary(label, cohort, original_count):
    """Display counts and prevalence for one explicitly named selection."""
    encounter_count = len(cohort)
    positive_count = (cohort["readmitted"] == "<30").sum()
    prevalence = (
        positive_count / encounter_count * 100
        if encounter_count > 0 else float("nan")
    )
    print(
        f"{label}: {encounter_count:,} encounters | "
        f"<30: {positive_count:,} ({prevalence:.2f}%) | "
        f"Excluded: {original_count - encounter_count:,}"
    )


def main():
    # Read source strings without interpreting missing-value markers automatically.
    data = pd.read_csv(
        "data/raw/diabetic_data.csv",
        dtype=str,
        keep_default_na=False,
    )

    print_section("DATA AUDIT | Original CSV")
    print(f"Encounters: {len(data):,} | Columns: {len(data.columns)}")

    # Profile explicit markers; blank counts include exact empty strings.
    question_mark_counts = (data == "?").sum()
    question_mark_percentages = question_mark_counts / len(data) * 100
    empty_string_counts = (data == "").sum()

    blank_counts = {}

    for column in data.columns:
        blank_counts[column] = (data[column].str.strip() == "").sum()

    missing_profile = pd.DataFrame(
        {
            "question_mark_count": question_mark_counts,
            "question_mark_percentage": question_mark_percentages,
            "empty_string_count": empty_string_counts,
            "blank_count": pd.Series(blank_counts),
        }
    )

    missing_profile = missing_profile.sort_values(
        "question_mark_count",
        ascending=False,
    )

    print_section("1. Explicit missing markers")
    visible_profile = missing_profile.loc[
        (missing_profile["question_mark_count"] > 0)
        | (missing_profile["empty_string_count"] > 0)
        | (missing_profile["blank_count"] > 0)
    ]
    print(
        visible_profile.to_string(
            formatters={"question_mark_percentage": lambda value: f"{value:.2f}%"}
        )
    )
    print(
        f"Showing {len(visible_profile)} of {len(missing_profile)} columns; "
        "remaining columns have zero for these markers."
    )
    print("Blank counts include empty strings; do not add them together.")

    print_section("1b. Recorded gender categories")
    print(data["gender"].value_counts(dropna=False).to_string())

    # Inspect documented laboratory categories, preserving nonmeasurement as text.
    allowed_lab_values = {
        "max_glu_serum": {"None", "Norm", ">200", ">300"},
        "A1Cresult": {"None", "Norm", ">7", ">8"},
    }

    print_section("2. Laboratory categories | 'None' means not measured")
    for column, allowed_values in allowed_lab_values.items():
        print(f"\n{column}")
        print(data[column].value_counts(dropna=False).to_string())

        not_measured_count = (data[column] == "None").sum()
        not_measured_percentage = not_measured_count / len(data) * 100
        print(
            f"Not measured: {not_measured_count} "
            f"({not_measured_percentage:.2f}%)"
        )

        observed_values = set(data[column].unique())
        unexpected_values = observed_values - allowed_values

        print("Unexpected category count:", len(unexpected_values))

    # Audit stay duration: numeric, integer and within the documented 1-14 days.
    hospital_stay_days = pd.to_numeric(
        data["time_in_hospital"],
        errors="coerce",
    )

    print_section("3. Hospital stay | Expected integer range: 1-14 days")
    print(f"Observed range: {hospital_stay_days.min():g}-{hospital_stay_days.max():g} days")
    print("Non-numeric stay values:", hospital_stay_days.isna().sum())

    stays_outside_range = (
        (hospital_stay_days < 1)
        | (hospital_stay_days > 14)
    )

    print("Stay values outside expected range:", stays_outside_range.sum())

    # Exclude conversion failures: they are counted separately, not as fractions.
    non_integer_stays = (hospital_stay_days.dropna() % 1) != 0
    print("Non-integer stay values:", non_integer_stays.sum())

    print_section("4. Numeric counts | Observed extrema, not validity limits")
    print(
        f"{'Field':<26} | {'Min':>7} | {'Max':>7} | "
        f"{'Non-numeric':>11} | {'Fractional':>10} | {'Negative':>8}"
    )
    audit_numeric_counts(data["num_lab_procedures"], "lab procedures")
    audit_numeric_counts(data["num_procedures"], "procedures")
    audit_numeric_counts(data["num_medications"], "medications")
    audit_numeric_counts(data["number_outpatient"], "prior outpatient visits")
    audit_numeric_counts(data["number_emergency"], "prior emergency visits")
    audit_numeric_counts(data["number_inpatient"], "prior inpatient encounters")
    audit_numeric_counts(data["number_diagnoses"], "diagnoses")

    prior_emergency_visits = pd.to_numeric(
        data["number_emergency"],
        errors="coerce",
    )

    print_section("5. Prior emergency visits | Distribution across encounters")
    print(
        prior_emergency_visits.describe(
            percentiles=[0.5, 0.95, 0.99]
        ).to_string(float_format=lambda value: f"{value:,.2f}")
    )

    above_three_count = (prior_emergency_visits > 3).sum()
    maximum_visit_count = prior_emergency_visits.max()
    maximum_frequency = (
        prior_emergency_visits == maximum_visit_count
    ).sum()

    print(f"\nEncounters with more than 3 prior visits: {above_three_count:,}")
    print(
        f"Maximum: {maximum_visit_count:g} | "
        f"Encounters with that maximum: {maximum_frequency:,}"
    )

    print_section("6. Recorded discharge disposition codes")
    print(
        data["discharge_disposition_id"]
        .value_counts(dropna=False)
        .to_string()
    )

    print_section("6b. Discharge codes and recorded readmission")
    discharge_readmission_counts = pd.crosstab(
        data["discharge_disposition_id"],
        data["readmitted"],
    )

    print(
        discharge_readmission_counts.reindex(
            index=["11", "19", "20", "21", "13", "14", "2"],
            columns=["<30", ">30", "NO"],
            fill_value=0,
        )
    )

    # Keep the code-11-only selection as a diagnostic comparison, not the main rule.
    print_section("7. Cohort comparison | Source and staging remain unchanged")
    code_11_only_cohort = data.loc[
        data["discharge_disposition_id"] != "11"
    ].copy()
    analysis_cohort = select_analysis_cohort(data)
    print_cohort_summary("Original dataset", data, len(data))
    print_cohort_summary("Diagnostic: exclude code 11 only", code_11_only_cohort, len(data))
    print_cohort_summary("Working cohort: exclude 11/19/20/21", analysis_cohort, len(data))
    print("Working rule based on recorded labels; event timing remains a source limitation.")


    print_section("8. Check duplicates")
    comparison_columns = [col for col in data.columns if col != "encounter_id"]
    duplicate_mask = data.duplicated(
        subset=comparison_columns,
        keep=False,
    )
    print("Duplicate rows (all occurrences):", duplicate_mask.sum())


    print_section("9. Medication categories")
    medication_audit_columns = ["metformin", "insulin", "change", "diabetesMed"]
    for column_name in medication_audit_columns:
        print(data[column_name].value_counts(dropna=False).to_string())
        print()

    print_section("9b. Medication category check")
    medication_columns = data.loc[
        :, "metformin":"metformin-pioglitazone"
    ].columns
    allowed_medication_values = {"No", "Steady", "Up", "Down"}

    for column_name in medication_columns:
        observed_values = set(data[column_name])
        unexpected_values = observed_values - allowed_medication_values

        print(column_name)
        print("Number of observed categories:", len(observed_values))
        print("Unexpected categories:", sorted(unexpected_values))

        if len(observed_values) == 1:
            print(f"{column_name} is a constant column")
            print(data[column_name].value_counts(dropna=False).to_string())
        print()


    print_section("9c. Medication indicator checks")
    allowed_indicator_values = {
        "change": {"No", "Ch"},
        "diabetesMed": {"No", "Yes"},
    }

    for column_name, allowed_values in allowed_indicator_values.items():
        observed_values = set(data[column_name].unique())
        unexpected_values = observed_values - allowed_values

        print(f"{column_name} | Unexpected categories: {sorted(unexpected_values)}")


    print_section("10. Diagnosis codes")
    for column_name in ["diag_1", "diag_2", "diag_3"]:
        recorded_codes = data.loc[data[column_name] != "?", column_name]

        print(recorded_codes.value_counts().head(5).to_string())
        print("Number of distinct codes:", recorded_codes.nunique())
        print()


    # Export only the marker summary; numeric and lab audits remain console output.
    report_path = Path("docs/reports/missing_marker_profile.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing_profile.to_csv(report_path, index_label="feature")
    print_section("Export")
    print("Full missing-marker report:", report_path)
    print(f"Contains all {len(missing_profile)} columns; numeric/lab audits are console-only.")


if __name__ == "__main__":
    main()
