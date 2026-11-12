"""Original learner-written CSV aggregation exercise; not an operational command."""

import csv


from carepath_lab.validation import read_header, validate_and_extract_record


def main():
    with open("data/raw/diabetic_data.csv", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        columns = read_header(reader)

        print("Number of columns:", len(columns))
        print("Column names:", columns)

        readmitted_index = columns.index("readmitted")
        admission_type_id_index = columns.index("admission_type_id")
        age_index = columns.index("age")
        patient_nbr_index = columns.index("patient_nbr")
        encounter_id_index = columns.index("encounter_id")

        encounter_count = 0
        readmitted_counts = {"<30": 0, ">30": 0, "NO": 0}
        admission_type_id_counts = {}
        age_counts = {}
        early_readmission_counts_by_age = {}
        patient_nbrs = set()
        encounter_counts_by_patient = {}
        encounter_counts_by_id = {}

        for row in reader:
            encounter_id, patient_nbr, readmitted_value = validate_and_extract_record(
                row,
                columns,
                encounter_id_index,
                patient_nbr_index,
                readmitted_index,
                readmitted_counts,
            )

            admission_type_id = row[admission_type_id_index]
            age = row[age_index]

            encounter_count += 1
            readmitted_counts[readmitted_value] += 1
            admission_type_id_counts[admission_type_id] = (
                admission_type_id_counts.get(admission_type_id, 0) + 1
            )
            age_counts[age] = age_counts.get(age, 0) + 1

            if readmitted_value == "<30":
                early_readmission_counts_by_age[age] = (
                    early_readmission_counts_by_age.get(age, 0) + 1
                )

            patient_nbrs.add(patient_nbr)
            encounter_counts_by_patient[patient_nbr] = (
                encounter_counts_by_patient.get(patient_nbr, 0) + 1
            )

            encounter_counts_by_id[encounter_id] = (
                encounter_counts_by_id.get(encounter_id, 0) + 1
            )

        if encounter_count == 0:
            raise ValueError("CSV contains no encounter records.")

        print("Recorded encounters:", encounter_count)
        print("Readmitted counts:", readmitted_counts)
        counts_match = sum(readmitted_counts.values()) == encounter_count
        print("Readmitted counts match:", counts_match)

        for category, count in readmitted_counts.items():
            percentage = (count / encounter_count) * 100
            print(f"{category} percentage: {percentage:.2f}%")

        print("Admission type counts:", admission_type_id_counts)
        unknown_admission_count = (
            admission_type_id_counts.get("5", 0)
            + admission_type_id_counts.get("6", 0)
            + admission_type_id_counts.get("8", 0)
        )
        unknown_admission_percentage = (unknown_admission_count / encounter_count) * 100
        print("Unknown admission count:", unknown_admission_count)
        print(f"Unknown admission percentage: {unknown_admission_percentage:.2f}%")
        print("Encounter counts by age interval:", age_counts)
        print("Early readmission counts by age:", early_readmission_counts_by_age)
        early_readmission_percentage_by_age = {
            age: (
                early_readmission_counts_by_age.get(age, 0)
                / age_counts[age]
                * 100
            )
            for age in age_counts
        }

        for age, percentage in early_readmission_percentage_by_age.items():
            early_count = early_readmission_counts_by_age.get(age, 0)
            print(
                f"    {age}: {early_count}/{age_counts[age]} "
                f"({percentage:.2f}%)"
            )

        print("Distinct patients:", len(patient_nbrs))
        patients_with_multiple_encounters = 0
        for encounters in encounter_counts_by_patient.values():
            if encounters > 1:
                patients_with_multiple_encounters += 1
        print("Patients with multiple recorded encounters:", patients_with_multiple_encounters)

        percentage_patients_with_multiple_encounters = (
            patients_with_multiple_encounters / len(patient_nbrs) * 100
        )
        print(
            "Percentage of patients with multiple recorded encounters: "
            f"{percentage_patients_with_multiple_encounters:.2f}%"
        )

        repeated_encounter_id_count = 0
        for occurrences in encounter_counts_by_id.values():
            if occurrences > 1:
                repeated_encounter_id_count += 1
        print("Repeated encounter IDs:", repeated_encounter_id_count)

if __name__ == "__main__":
    main()
