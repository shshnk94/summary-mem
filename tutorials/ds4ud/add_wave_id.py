"""Add a `wave_id` column to `msg_essays_v9v11` disambiguating waves across people.

`year_wave` (e.g. 'y2w1') identifies a wave within a person, but collides across people --
every person shares the same `year_wave` labels. `wave_id` disambiguates it by prefixing
with `person_id`, giving `tutorials/prediction.py` a per-person-per-wave grouping key to
pass as `--conversation_field`, mapping ds4ud's (person -> wave -> EMA) hierarchy onto the
pipeline's (person -> conversation -> turn) one.

Added directly to `msg_essays_v9v11` (rather than a separate view/table) so every existing
`--message_table msg_essays_v9v11` invocation sees the column with no other change needed.
Column existence is checked via information_schema rather than `IF EXISTS`/`IF NOT EXISTS`
clauses, which aren't uniformly supported across MySQL/MariaDB versions.
"""

from sqlalchemy import create_engine, text

DATABASE = "ssubrahmanya"
TABLE = "msg_essays_v9v11"
COLUMN = "wave_id"


def build():
    engine = create_engine(
        f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4&read_default_file=~/.my.cnf"
    )
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = :db AND table_name = :table AND column_name = :column"
            ),
            {"db": DATABASE, "table": TABLE, "column": COLUMN},
        ).scalar()

        if exists:
            conn.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"))

        conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(80)"))
        conn.execute(text(f"UPDATE {TABLE} SET {COLUMN} = CONCAT(person_id, '_', year_wave)"))
        conn.execute(text(f"ALTER TABLE {TABLE} ADD INDEX ({COLUMN})"))
        conn.commit()  # DDL autocommits, but harmless to call; UPDATE needs it

    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(DISTINCT {COLUMN}) FROM {TABLE}")).scalar()

    print(f"[{TABLE}.{COLUMN}: {count} distinct waves]")


if __name__ == "__main__":
    build()
