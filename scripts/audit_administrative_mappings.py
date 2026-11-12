"""Audit administrative-code coverage and unavailable-information labels."""

import csv

import pandas as pd


def main():
    with open("data/raw/IDS_mapping.csv", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        current_section = None

        mappings = {
            "admission_type_id": {},
            "discharge_disposition_id": {},
            "admission_source_id": {},
        }

        for row in reader:
            if len(row) != 2:
                raise ValueError("Unexpected number of fields in mapping record.")

            if row == ["", ""]:
                current_section = None
                continue

            code = row[0]
            description = row[1]

            if code in mappings:
                if description != "description":
                    raise ValueError("Unexpected mapping section header.")

                current_section = code
                continue

            if current_section is None or code.strip() == "" or description.strip() == "":
                raise ValueError("Current section is None/Code is empty/Description is empty")

            if code in mappings[current_section]:
                raise ValueError("Duplicate code in mapping section.")
            mappings[current_section][code] = description

    data = pd.read_csv(
        "data/raw/diabetic_data.csv",
        usecols=list(mappings),
        dtype=str,
        keep_default_na=False,
    )

    unavailable_descriptions = {
        "Not Available",
        "NULL",
        "Not Mapped",
        "Unknown/Invalid",
    }

    for section_name, section_mapping in mappings.items():
        observed_codes = set(data[section_name].unique())
        mapped_codes = set(section_mapping)
        unmapped_codes = observed_codes - mapped_codes

        print(f"{section_name} has {len(observed_codes)} observed codes\n"
              f"Unmapped codes: {sorted(unmapped_codes)}")

        unavailable_codes = set()
        for code, description in section_mapping.items():
            if description.strip() in unavailable_descriptions:
                unavailable_codes.add(code)

        unavailable_mask = data[section_name].isin(unavailable_codes)
        unavailable_encounter_count = unavailable_mask.sum()
        unavailable_encounter_percentage = unavailable_encounter_count / len(data) * 100
        print(f"Rows with unavailable codes: {unavailable_encounter_count} ({unavailable_encounter_percentage:.2f}%)")


if __name__ == "__main__":
    main()
