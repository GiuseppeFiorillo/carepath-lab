import csv

import psycopg
from psycopg import sql

from carepath_lab.validation import read_header, validate_and_extract_record


def main():
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="carepath_lab",
        connect_timeout=5
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                    AND table_name = 'encounters_raw'
                ORDER BY ordinal_position;
                """
            )
            database_columns = cursor.fetchall()

            database_column_names = []
            for column_record in database_columns:
                database_column_names.append(column_record[0])

            with open("data/raw/diabetic_data.csv", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                csv_column_names = read_header(reader)

                if database_column_names != csv_column_names:
                    raise ValueError("CSV header does not match database columns.")

                cursor.execute("SELECT COUNT(*) FROM encounters_raw")
                result = cursor.fetchone()
                existing_row_count = result[0]

                if existing_row_count > 0:
                    raise ValueError("Raw table must be empty before loading.")

                column_identifiers = []
                for column_name in csv_column_names:
                    column_identifiers.append(sql.Identifier(column_name))

                insert_columns = sql.SQL(", ").join(column_identifiers)

                value_placeholders = [sql.Placeholder() for _ in csv_column_names]
                insert_placeholders = sql.SQL(", ").join(value_placeholders)

                insert_query = sql.SQL(
                    "INSERT INTO encounters_raw ({}) VALUES ({});"
                ).format(insert_columns, insert_placeholders)


                encounter_id_index = csv_column_names.index("encounter_id")
                patient_nbr_index = csv_column_names.index("patient_nbr")
                readmitted_index = csv_column_names.index("readmitted")

                allowed_readmitted_values = {"<30", ">30", "NO"}
                loaded_count = 0

                for row in reader:
                    validate_and_extract_record(
                        row,
                        csv_column_names,
                        encounter_id_index,
                        patient_nbr_index,
                        readmitted_index,
                        allowed_readmitted_values
                    )
                    cursor.execute(insert_query, row)
                    loaded_count += 1

        if loaded_count == 0:
            raise ValueError("CSV contains no encounter records.")

    print("Loaded encounters:", loaded_count)


if __name__ == "__main__":
    main()
