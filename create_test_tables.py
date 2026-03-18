"""
create_test_tables.py
=====================
reads directly from the MySQL training table (ml_model.ai412024) and 
creates 6 separate test tables in the same database:

    test_normal   — rows where `Machine failure` = 0  (healthy operation)
    test_twf      — rows where TWF = 1                (Tool-Wear Failure)
    test_hdf      — rows where HDF = 1                (Heat-Dissipation Failure)
    test_pwf      — rows where PWF = 1                (Power Failure)
    test_osf      — rows where OSF = 1                (Over-Strain Failure)
    test_rnf      — rows where RNF = 1                (Random Failure)

Each test table contains ONLY the 5 raw sensor columns the predictor needs:
    Air temperature [K]
    Process temperature [K]
    Rotational speed [rpm]
    Torque [Nm]
    Tool wear [min]

Usage:
    python create_test_tables.py

Safe to re-run as well — checks if test tables exist; if so, skips creation/population.
"""

import sys
from sqlalchemy import create_engine, text

# ── Config ─────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "ml_model",
}

SOURCE_TABLE = "ai412024"  # the training table you already uploaded

# Each entry: (dest_table, WHERE clause referencing source table columns)
TABLES = [
    ("test_normal", "`Machine failure` = 0"),
    ("test_twf", "TWF = 1"),
    ("test_hdf", "HDF = 1"),
    ("test_pwf", "PWF = 1"),
    ("test_osf", "OSF = 1"),
    ("test_rnf", "RNF = 1"),
]

# DDL for every test table — just an id + the 5 sensor columns
CREATE_SQL = """
    CREATE TABLE `{dest}` (
        id                          INT AUTO_INCREMENT PRIMARY KEY,
        `Air temperature [K]`       FLOAT NOT NULL,
        `Process temperature [K]`   FLOAT NOT NULL,
        `Rotational speed [rpm]`    FLOAT NOT NULL,
        `Torque [Nm]`               FLOAT NOT NULL,
        `Tool wear [min]`           FLOAT NOT NULL
    )
"""

# Populate directly inside MySQL — no Python-side data transfer needed
POPULATE_SQL = """
    INSERT INTO `{dest}` (
        `Air temperature [K]`,
        `Process temperature [K]`,
        `Rotational speed [rpm]`,
        `Torque [Nm]`,
        `Tool wear [min]`
    )
    SELECT
        `Air temperature [K]`,
        `Process temperature [K]`,
        `Rotational speed [rpm]`,
        `Torque [Nm]`,
        `Tool wear [min]`
    FROM `{src}`
    WHERE {where}
"""

# ── Main ────────────────────────────────────────────────────────────────────────


def check_table_exists(conn, table_name):
    """
    Checks if a table exists in the current database using the SHOW TABLES command.
    """
    result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'"))
    return result.fetchone() is not None


def main():
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
    print(
        f"Connecting to MySQL: {DB_CONFIG['user']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
    try:
        engine = create_engine(url)
        print("  → Engine created OK\n")
    except Exception as e:
        print(f"ERROR: Could not create SQLAlchemy engine — {e}")
        sys.exit(1)

    with engine.connect() as conn:
        # Verify the source training table exists
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = :db AND table_name = :table"
            ),
            {"db": DB_CONFIG["database"], "table": SOURCE_TABLE},
        )

        if result.scalar() == 0:
            print(
                f"ERROR: Source table '{SOURCE_TABLE}' not found in '{DB_CONFIG['database']}'."
            )
            return

        result = conn.execute(text(f"SELECT COUNT(*) FROM `{SOURCE_TABLE}`"))
        total_rows = result.scalar()
        print(f"Source table '{SOURCE_TABLE}': {total_rows:,} rows\n")

        # Create and populate each test table
        for dest_table, where_clause in TABLES:
            print(f"[{dest_table}]")

            if check_table_exists(conn, dest_table):
                print(f"  → Already exists, skipping creation.")
                continue

            # Create fresh table
            conn.execute(text(CREATE_SQL.format(dest=dest_table)))
            print(f"  → Table created")

            # Populate with INSERT ... SELECT directly in MySQL
            conn.execute(
                text(
                    POPULATE_SQL.format(
                        dest=dest_table,
                        src=SOURCE_TABLE,
                        where=where_clause,
                    )
                )
            )
            conn.commit()

            result = conn.execute(text(f"SELECT COUNT(*) FROM `{dest_table}`"))
            inserted = result.scalar()
            print(f"  → {inserted:,} rows inserted\n")

    print("Test tables created successfully:\n")
    for dest_table, where_clause in TABLES:
        print(f"  • {dest_table:<14}  (WHERE {where_clause})")


if __name__ == "__main__":
    main()
