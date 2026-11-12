"""Verify that encounters_raw matches the source CSV without modifying either.

Compare column names/order and all field values by encounter_id, independently
of row order. Report missing, unexpected and mismatched records; raise ValueError
if differences are found.
"""

import csv

import psycopg
from psycopg import sql

from carepath_lab.validation import read_header, validate_and_extract_record


def main():
    with open("data/raw/diabetic_data.csv", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        columns = read_header(reader)

        encounter_id_index = columns.index("encounter_id")
        patient_nbr_index = columns.index("patient_nbr")
        readmitted_index = columns.index("readmitted")

        allowed_readmitted_values = {"<30", ">30", "NO"}
        source_records = {}

        for row in reader:
            encounter_id, _, _ = validate_and_extract_record(
                row,
                columns,
                encounter_id_index,
                patient_nbr_index,
                readmitted_index,
                allowed_readmitted_values,
            )

            if encounter_id in source_records:
                raise ValueError("Duplicate encounter ID in source CSV.")

            source_records[encounter_id] = tuple(row)

        print("Source encounters:", len(source_records))


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

            if database_column_names != columns:
                raise ValueError("CSV header does not match database columns.")
            print("CSV header matches database columns.")

            column_identifiers = []
            for column_name in columns:
                column_identifiers.append(sql.Identifier(column_name))

            select_columns = sql.SQL(", ").join(column_identifiers)
            select_query = sql.SQL(
                "SELECT {} FROM encounters_raw;"
            ).format(select_columns)

            cursor.execute(select_query)

            database_count = 0
            unexpected_record_count = 0
            mismatched_record_count = 0

            for database_row in cursor:
                encounter_id = database_row[encounter_id_index]
                source_row = source_records.pop(encounter_id, None)

                database_count += 1
                if source_row is None:
                    unexpected_record_count += 1
                elif source_row != database_row:
                    mismatched_record_count += 1

            missing_record_count = len(source_records)

            print("Database encounters:", database_count)
            print("Missing records:", missing_record_count)
            print("Unexpected records:", unexpected_record_count)
            print("Mismatched records:", mismatched_record_count)

            if unexpected_record_count != 0 or mismatched_record_count != 0 or missing_record_count != 0:
                raise ValueError("Database records do not match source CSV.")
            print("Raw import verified: all source field values match.")

if __name__ == "__main__":
    main()
