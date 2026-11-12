"""Demonstrate rollback with synthetic rows in a temporary PostgreSQL table."""

import psycopg


def main():
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="carepath_lab",
        connect_timeout=5,
        autocommit=True
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE rollback_demo (example_id TEXT PRIMARY KEY)"
            )

            try:
                with connection.transaction():
                    cursor.execute(
                        "INSERT INTO rollback_demo VALUES (%s)",
                        ("synthetic-A",)
                    )
                    raise ValueError("Simulated validation failure.")
            except ValueError:
                print("Expected error caught.")

            cursor.execute("SELECT COUNT(*) FROM rollback_demo;")
            print("Rows after rollback:", cursor.fetchone()[0])


if __name__ == "__main__":
    main()
