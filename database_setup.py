"""
database_setup.py — ANPR System Database Initialiser
=====================================================
Creates the SQLite database `anpr_system.db` with two tables:
  • owners  — maps plate numbers to owner names and vehicle models.
  • logs    — time-stamped detection log with confidence scores.

Run this script ONCE before launching the pipeline:
    python database_setup.py

Optimisations for Edge (Raspberry Pi):
  • WAL journal mode   → allows concurrent reads while the pipeline writes.
  • NORMAL synchronous  → reduces fsync calls; safe enough for logging data.
  • Page size 4096      → matches typical SD-card block size.
"""

import sqlite3
import os
import sys

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
DB_NAME = "anpr_system.db"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)


def create_database() -> None:
    """Create the ANPR database and seed it with dummy owner records."""

    conn: sqlite3.Connection | None = None

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # ----- PRAGMA tuning for low-power embedded devices -----
        cursor.execute("PRAGMA journal_mode = WAL;")       # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous  = NORMAL;")    # Reduce disk I/O
        cursor.execute("PRAGMA page_size    = 4096;")      # SD-card-friendly

        # -----------------------------------------------------------------
        #  Table: owners
        #  Stores vehicle owner information linked by plate_number (PK).
        # -----------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS owners (
                plate_number  TEXT PRIMARY KEY NOT NULL,
                owner_name    TEXT NOT NULL,
                vehicle_model TEXT NOT NULL
            );
        """)

        # -----------------------------------------------------------------
        #  Table: logs
        #  Each row is a single plate detection event.
        #  log_id is auto-incremented; plate_number is a FK to owners.
        # -----------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number     TEXT    NOT NULL,
                entry_date       TEXT    NOT NULL,
                entry_time       TEXT    NOT NULL,
                confidence_score REAL    NOT NULL,
                FOREIGN KEY (plate_number) REFERENCES owners (plate_number)
            );
        """)

        # Index on plate_number in logs for faster lookups during
        # duplicate filtering and owner-name JOINs.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_plate
            ON logs (plate_number);
        """)

        # -----------------------------------------------------------------
        #  Seed data — 5 dummy owner records for testing
        # -----------------------------------------------------------------
        dummy_owners = [
            ("MH12AB1234", "Gaurav Srivastava",  "Maruti Suzuki Swift"),
            ("DL01CD5678", "Ananya Sharma",       "Hyundai Creta"),
            ("KA03EF9012", "Rohan Mehta",         "Tata Nexon EV"),
            ("TN07GH3456", "Priya Iyer",          "Honda City"),
            ("UP16JK7890", "Vikram Singh",        "Mahindra Thar"),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO owners (plate_number, owner_name, vehicle_model)
            VALUES (?, ?, ?);
            """,
            dummy_owners,
        )

        conn.commit()
        print(f"[✓] Database created successfully at: {DB_PATH}")
        print(f"[✓] Seeded {len(dummy_owners)} dummy owner records.")

    except sqlite3.Error as exc:
        print(f"[✗] SQLite error during setup: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    create_database()
