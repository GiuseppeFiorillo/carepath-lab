"""Optional first-section mapping loader for the original SQL join exercise."""

import csv

import psycopg


def main():
    with open("data/raw/IDS_mapping.csv", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        columns = next(reader, None)

        if columns != ["admission_type_id", "description"]:
            raise ValueError("Unexpected admission mapping header.")

        mapping_rows = []

        for row in reader:
            if row == ["", ""]:
                break

            if len(row) != 2:
                raise ValueError("Unexpected length of the row.")

            if row[0].strip() == "" or row[1].strip() == "":
                raise ValueError("Code or description cannot be empty.")

            mapping_rows.append(row)

        if not mapping_rows:
            raise ValueError("Admission mapping contains no records.")

        print("Admission mapping rows:", len(mapping_rows))

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="carepath_lab",
        connect_timeout=5
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM admission_type_mapping")
            result = cursor.fetchone()
            existing_row_count = result[0]

            if existing_row_count > 0:
                raise ValueError("Admission type mapping table must be empty before loading.")

            cursor.executemany(
                """
                INSERT INTO admission_type_mapping (
                    admission_type_id,
                    description
                )
                VALUES (%s, %s)
                """,
                mapping_rows
            )

    print("Admission type mapping table successfully filled.")

if __name__ == "__main__":
    main()
