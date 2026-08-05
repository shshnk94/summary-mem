"""Build `outcomes_v9v11_person`, the person-level aggregate of `outcomes_v9v11`.

`outcomes_v9v11` is a wave-level table: multiple `year_wave` rows per `person_id`
(e.g. y2w1..y2w5), one per EMA/diary assessment. `outcomes_v9v11_person` averages
the five OCEAN scores across those rows per `person_id` -- the table
`train_model.py`'s ds4ud dataset trains against.

`cohort` is constant per `person_id` in the source table, so it carries over
unchanged rather than being aggregated. `anxiety_score`, `depression_score`,
`phq9_sum`, and `gad7_sum` aren't part of the person-level table and are dropped.
"""

from sqlalchemy import create_engine, text

DATABASE = "ssubrahmanya"
SOURCE_TABLE = "outcomes_v9v11"
TARGET_TABLE = "outcomes_v9v11_person"

SCORE_COLUMNS = [
    "openness_score",
    "conscientious_score",
    "extravert_score",
    "agreeable_score",
    "neurotic_score",
]


def build():
    engine = create_engine(
        f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4&read_default_file=~/.my.cnf"
    )
    with engine.connect() as conn:

        conn.execute(text(f"DROP TABLE IF EXISTS {TARGET_TABLE}"))

        score_columns = ", ".join(f"{c} DOUBLE" for c in SCORE_COLUMNS)
        conn.execute(
            text(
                f"CREATE TABLE {TARGET_TABLE} ("
                f"  person_id VARCHAR(64) NOT NULL PRIMARY KEY,"
                f"  cohort VARCHAR(10) NOT NULL,"
                f"  {score_columns}"
                f")"
            )
        )

        avg_select = ", ".join(f"AVG({c}) AS {c}" for c in SCORE_COLUMNS)
        inserted = conn.execute(
            text(
                f"INSERT INTO {TARGET_TABLE} (person_id, cohort, {', '.join(SCORE_COLUMNS)}) "
                f"SELECT person_id, MIN(cohort), {avg_select} "
                f"FROM {SOURCE_TABLE} "
                f"GROUP BY person_id"
            )
        )
        conn.commit()  # DDL autocommits, the INSERT does not

    print(f"[{TARGET_TABLE}: {inserted.rowcount} people]")


if __name__ == "__main__":
    build()
