"""Differential Language Analysis (DLA) over summary-mem's rolling summaries.

Follows https://dlatk.github.io/dlatk/tutorials/tut_dla.html, but the corpus is each
speaker's rolling summary (SummaryMemory's `summaries` or `summaries_plain` table in
`--summary_db`) rather than raw corpus text -- so this asks
which 1-to-3 grams in *those summaries* differentiate a per-speaker outcome (e.g. a
personality trait), grouped at the person level (`-c speaker_id`).

Two steps, either runnable on its own (no step flag runs both):
  --extract     copy the summaries into a DLATK-shaped message table and build its
                occurrence-filtered 1to3gram feature table.
  --correlate   correlate the filtered features with --outcomes and render
                positive/negative word clouds per outcome.

`--dataset` supplies the per-corpus outcome_table/outcome_key/outcomes preset -- the
summary text itself always comes from `--summary_db`/`--summary_table`, since
SummaryMemory writes there regardless of which corpus it summarized.

Usage:
    python dla.py --dataset candor --extract
    python dla.py --dataset candor --correlate
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import Double, String, create_engine, text

### ---- Configuration ----

# Per-corpus outcome presets, selected by --dataset (any field still individually
# overridable). outcome_key is the *outcome table's* person column, renamed to
# speaker_id before use -- the summaries table always groups by speaker_id already,
# whatever the raw corpus called it.
DATASETS = {
    "candor": dict(
        database="candor",
        outcome_table="surveys",
        outcome_key="user_id",
        outcomes=["my_open", "my_conscientious", "my_extraversion", "my_agreeable", "my_neurotic"],
    ),
    "ds4ud": dict(
        database="ssubrahmanya",
        outcome_table="outcomes_v9v11_person",
        outcome_key="person_id",
        outcomes=[
            "openness_score",
            "conscientious_score",
            "extravert_score",
            "agreeable_score",
            "neurotic_score",
        ],
    ),
    "2m2w": dict(
        database="ssubrahmanya",
        outcome_table="outcomes_user_2m2w",
        outcome_key="user_id",
        outcomes=[
            "avg_from_wave_openness_score",
            "avg_from_wave_conscientious_score",
            "avg_from_wave_extravert_score",
            "avg_from_wave_agreeable_score",
            "avg_from_wave_neurotic_score",
        ],
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"

### ---- Execution ----


def make_engine(database: str):
    return create_engine(
        f"mysql://ssubrahmanya@localhost/{database}?charset=utf8mb4&read_default_file=~/.my.cnf"
    )


def build_message_table(summary_engine, summary_table: str, message_table: str) -> None:
    """STEP 0 of the tutorial: copy `summary_table` into a table DLATK can group by.

    `summary_table` has a composite (conversation_id, speaker_id) primary key and no
    single-column message id, so DLATK's --message_field/--messageid_field can't point
    at it directly. Copying into DLATK's own default column names (`message`,
    auto_increment `message_id`) means every dlatkInterface.py call below can omit
    those flags entirely.
    """
    with summary_engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {message_table}"))
        conn.execute(
            text(
                f"""
                CREATE TABLE {message_table} (
                    message_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    speaker_id VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    KEY (speaker_id)
                ) ENGINE=InnoDB
                """
            )
        )
        conn.execute(
            text(f"INSERT INTO {message_table} (speaker_id, message) SELECT speaker_id, summary FROM {summary_table}")
        )
        count = conn.execute(text(f"SELECT COUNT(DISTINCT speaker_id) FROM {message_table}")).scalar()
    print(f"[{message_table}: {count} speakers]")


def extract_ngrams(summary_db: str, message_table: str, feat_table: str, filtered_feat_table: str, group_freq_thresh: int) -> None:
    """Step 1: 1-to-3 gram feature table, occurrence-filtered.

    Mirrors the DLA tutorial's Step 1 exactly: extract each n, combine into 1to3gram,
    then drop n-grams used by less than 5% of speakers (--set_p_occ). --group_freq_thresh
    only applies to the *filter* call -- it needs the per-speaker word-count table
    --add_ngrams just built.
    """
    command = [
        PYTHON, DLATK,
        "-d", summary_db,
        "-t", message_table,
        "-c", "speaker_id",
        "--add_ngrams",
        "-n", "1", "2", "3",
        "--combine_feat_tables", "1to3gram",
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here

    command = [
        PYTHON, DLATK,
        "-d", summary_db,
        "-t", message_table,
        "-c", "speaker_id",
        "-f", feat_table,
        "--feat_occ_filter",
        "--set_p_occ", "0.05",
        "--group_freq_thresh", str(group_freq_thresh),
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    print(f"[features -> {summary_db}.{filtered_feat_table}]")


def prepare_outcome_table(
    outcome_engine, summary_engine, message_table: str, outcome_table_name: str,
    source_outcome_table: str, outcome_key: str, outcomes: list[str],
) -> None:
    """Rebuild `outcome_table_name` in the summary database: one outcome vector per
    speaker who has a summary, averaged over duplicate outcome rows (e.g. CANDOR
    surveys a person once per conversation).

    `outcome_engine` and `summary_engine` may point at different MySQL databases (e.g.
    CANDOR's surveys vs. the summaries store); pandas, not a SQL join, bridges them.
    """
    speaking = pd.read_sql(f"SELECT DISTINCT speaker_id FROM {message_table}", summary_engine)
    speaking["speaker_id"] = speaking["speaker_id"].astype(str)

    raw = pd.read_sql(f"SELECT {outcome_key}, {', '.join(outcomes)} FROM {source_outcome_table}", outcome_engine)
    raw = raw.rename(columns={outcome_key: "speaker_id"})
    raw["speaker_id"] = raw["speaker_id"].astype(str)

    labelled = raw.groupby("speaker_id", as_index=False)[outcomes].mean()
    labelled = labelled.dropna(subset=outcomes)
    labelled = labelled[labelled["speaker_id"].isin(speaking["speaker_id"])]

    with summary_engine.begin() as conn:
        labelled.to_sql(
            outcome_table_name, conn, if_exists="replace", index=False,
            dtype={"speaker_id": String(255), **{o: Double for o in outcomes}},
        )
        conn.execute(text(f"ALTER TABLE {outcome_table_name} ADD PRIMARY KEY (speaker_id)"))
    print(f"[{outcome_table_name}: {len(labelled)} of {len(speaking)} speakers labelled]")


def correlate_wordclouds(
    summary_db: str, message_table: str, filtered_feat_table: str, outcome_table_name: str,
    outcomes: list[str], group_freq_thresh: int, output_dir: Path, output_name: str,
) -> None:
    """Step 2: correlate the filtered 1to3grams with --outcomes and render
    positive/negative word clouds per outcome (DLA tutorial's Step 2)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_name
    command = [
        PYTHON, DLATK,
        "-d", summary_db,
        "-t", message_table,
        "-c", "speaker_id",
        "-f", filtered_feat_table,
        "--outcome_table", outcome_table_name,
        "--outcomes", *outcomes,
        "--group_freq_thresh", str(group_freq_thresh),
        "--output_name", str(stem),
        "--tagcloud",
        "--make_wordclouds",
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    print(f"[wordclouds -> {stem}_tagcloud_wordclouds/]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), required=True,
        help="corpus preset for --database/--outcome_table/--outcome_key/--outcomes "
             "(each still individually overridable)",
    )
    parser.add_argument("--database", default=None, help="MySQL database holding the outcome table (default: --dataset preset)")
    parser.add_argument("--outcome_table", default=None, help="table holding one outcome vector per --outcome_key (default: --dataset preset)")
    parser.add_argument(
        "--outcome_key", default=None,
        help="outcome_table's person column, if named differently from speaker_id (default: --dataset preset)",
    )
    parser.add_argument(
        "--summary_db", default="ssubrahmanya",
        help="MySQL database SummaryMemory wrote rolling summaries to (default: %(default)s)",
    )
    parser.add_argument(
        "--summary_table", choices=["summaries", "summaries_plain"], default="summaries",
        help="which SummaryMemory table to run DLA over (default: %(default)s)",
    )
    parser.add_argument(
        "--group_freq_thresh", type=int, default=0,
        help="DLATK's minimum word count per speaker to be kept, applied after ngram "
             "extraction (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="where --correlate writes the tagcloud text/wordcloud images (default: %(default)s)",
    )
    parser.add_argument(
        "--output_name", default=None,
        help="stem for --correlate's output files (default: dla_<dataset>_<summary_table>)",
    )
    parser.add_argument("--extract", action="store_true", help="Step 1: build & occurrence-filter the 1to3gram feature table")
    parser.add_argument("--correlate", action="store_true", help="Step 2: correlate features with --outcomes and render word clouds")
    args = parser.parse_args()

    preset = DATASETS[args.dataset]
    args.database = args.database or preset["database"]
    args.outcome_table = args.outcome_table or preset["outcome_table"]
    args.outcome_key = args.outcome_key or preset["outcome_key"]
    outcomes = preset["outcomes"]

    # No step flag means run both.
    if not (args.extract or args.correlate):
        args.extract = args.correlate = True

    message_table = f"dla_{args.summary_table}"
    feat_table = f"feat$1to3gram${message_table}$speaker_id"
    filtered_feat_table = f"{feat_table}$0_05"
    outcome_table_name = f"dla_outcomes_{args.dataset}_{args.summary_table}"
    output_name = args.output_name or f"dla_{args.dataset}_{args.summary_table}"

    summary_engine = make_engine(args.summary_db)

    if args.extract:
        build_message_table(summary_engine, args.summary_table, message_table)
        extract_ngrams(args.summary_db, message_table, feat_table, filtered_feat_table, args.group_freq_thresh)

    if args.correlate:
        outcome_engine = summary_engine if args.database == args.summary_db else make_engine(args.database)
        prepare_outcome_table(
            outcome_engine, summary_engine, message_table, outcome_table_name,
            args.outcome_table, args.outcome_key, outcomes,
        )
        correlate_wordclouds(
            args.summary_db, message_table, filtered_feat_table, outcome_table_name,
            outcomes, args.group_freq_thresh, args.output_dir, output_name,
        )
