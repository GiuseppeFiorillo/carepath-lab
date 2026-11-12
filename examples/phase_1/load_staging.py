import csv

import psycopg

from carepath_lab.validation import read_header, validate_and_extract_record

INSERT_ENCOUNTER_SQL = """
INSERT INTO encounters_staging (
    encounter_id,
    patient_nbr,
    readmitted,
    admission_type_id,
    age
)
VALUES (%s, %s, %s, %s, %s);
"""


def main():
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="carepath_lab",
        connect_timeout=5
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM encounters_staging;")
            result = cursor.fetchone()
            existing_row_count = result[0]

            if existing_row_count > 0:
                raise ValueError("Staging table must be empty before loading.")

            with open("data/raw/diabetic_data.csv", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                columns = read_header(reader)

                readmitted_index = columns.index("readmitted")
                admission_type_id_index = columns.index("admission_type_id")
                age_index = columns.index("age")
                patient_nbr_index = columns.index("patient_nbr")
                encounter_id_index = columns.index("encounter_id")

                allowed_readmitted_values = {"<30", ">30", "NO"}

                loaded_count = 0

                for row in reader:
                    encounter_id, patient_nbr, readmitted_value = validate_and_extract_record(
                        row,
                        columns,
                        encounter_id_index,
                        patient_nbr_index,
                        readmitted_index,
                        allowed_readmitted_values
                    )

                    admission_type_id = row[admission_type_id_index]
                    age = row[age_index]

                    record_values = (
                        encounter_id,
                        patient_nbr,
                        readmitted_value,
                        admission_type_id,
                        age
                    )

                    cursor.execute(INSERT_ENCOUNTER_SQL, record_values)
                    loaded_count += 1

                if loaded_count == 0:
                    raise ValueError("CSV contains no encounter records.")

    print("Loaded encounters:", loaded_count)

if __name__ == "__main__":
    main()
