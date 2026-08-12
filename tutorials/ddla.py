"""Stage 1 DDLA: summary versus not-summary.

Per the advisor's plan: "Stage 1 DLA: summary versus not summary." Follows
https://dlatk.github.io/dlatk/tutorials/tut_dla.html, generalizing the
stack-two-corpora-with-a-binary-label trick from personallm's
commands/dla.py --ddla (there: human vs. model text; here: a person's
*summary* vs. their *full, un-summarized* language).

"Summary" is one memory mechanism's stored MySQL text -- SummaryMemory
(`summaries_plain`), BatchSummaryMemory (`summary_batch`), or
ConversationBatchSummaryMemory (`summary_conversation_batch`), selected via
--memory and scoped to one run via --namespace (those tables are shared
across every run that ever wrote to them). "Not summary" is that same
person's full in-context language -- every turn they ever produced,
concatenated exactly as InContextMemory hands it to prediction.py's
personality-assessment prompt (--memory in_context), minus the "Excerpt N:"
prompt scaffolding since that text isn't part of the person's language.

Rows are grouped at `message_id` (one row per person x representation, so a
person contributes two rows here) rather than person_id, since person_id
alone can't be a primary key once both representations share it.

Two steps, either runnable on its own (no step flag runs both):
  --extract     stack the two representations into one message table tagged
                with `is_summary`, and build its occurrence-filtered
                1to3gram feature table.
  --correlate   correlate the filtered features with `is_summary` and
                render positive ("more summary-like")/negative ("more
                full-text-like") word clouds -- Stage 2 of the plan later
                restricts to just the terms flagged here (not implemented
                by this script).

Usage:
    python ddla.py --namespace ds4ud_all_gpt4omini --memory summary --extract
    python ddla.py --namespace ds4ud_all_gpt4omini --memory summary --correlate
    python ddla.py --namespace ds4ud_all_gpt51 --memory batch_summary
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import Integer, String, Text, create_engine, text

### ---- Configuration ----

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"

GROUP_FIELD = "message_id"  # one row per document, so it's both DLATK's message_id and its group_id
LABEL_FIELD = "is_summary"

DATABASE = "ssubrahmanya"
MESSAGE_TABLE = "msg_essays_v9v11"
CONVERSATION_FIELD = "wave_id"
USER_FIELD = "person_id"
TURN_FIELD = "startdate"

# --memory -> MySQL table it reads from, mirroring summary_mem.memory's
# SummaryMemory/BatchSummaryMemory/ConversationBatchSummaryMemory constructors.
# mem0/raptor aren't MySQL summary tables in this same (namespace, speaker_id[,
# conversation_id], summary) shape, so they're not supported here.
SUMMARY_TABLES = {
    "summary": "summaries_plain",
    "batch_summary": "summary_batch",
    "conversation_batch_summary": "summary_conversation_batch",
}

### ---- Execution ----


def slug(name: str) -> str:
    """MySQL-identifier-safe stand-in for a namespace/memory name. Strips
    underscores too (not just non-alphanumerics) -- feat table names already
    stack DLATK's $-delimited naming convention (feat$1to3gram$<table>$<group>
    $<set_p_occ>) on top of this, so a verbose slug pushes the filtered feat
    table name past MySQL's 64-character identifier limit.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "", name).lower()


def make_engine(database: str):
    return create_engine(
        f"mysql://ssubrahmanya@localhost/{database}?charset=utf8mb4&read_default_file=~/.my.cnf"
    )


def read_full_text(
    engine, message_table: str, conversation_field: str, user_field: str, turn_field: str,
) -> pd.DataFrame:
    """A person's full, un-summarized language: every turn they produced,
    joined in order within a conversation ("\\n"), then across their
    conversations ("\\n\\n") -- the same substance InContextMemory.recall_speaker
    hands to the personality-assessment prompt, minus its "Excerpt N:" headers
    (prompt scaffolding, not language). Returns [user_field, message].
    """
    query = text(
        f"SELECT {user_field}, {conversation_field}, message FROM {message_table} "
        f"WHERE message IS NOT NULL "
        f"ORDER BY {user_field}, {conversation_field}, {turn_field}"
    )
    turns = pd.read_sql(query, engine)
    turns[user_field] = turns[user_field].astype(str)

    conversations = turns.groupby([user_field, conversation_field])["message"].apply("\n".join)
    full_text = conversations.groupby(user_field).apply("\n\n".join)
    return full_text.reset_index(name="message")


def read_summary_text(
    engine, memory: str, summary_table: str, namespace: str, user_field: str,
) -> pd.DataFrame:
    """A person's stored summary text for one --memory mechanism, scoped to
    --namespace. BatchSummaryMemory's table already holds one row per speaker;
    SummaryMemory/ConversationBatchSummaryMemory's holds one row per
    (conversation, speaker), joined here across a speaker's conversations
    ("\\n\\n"), mirroring their own recall_speaker(). Returns [user_field, message].
    """
    if memory == "batch_summary":
        query = text(f"SELECT speaker_id, summary FROM {summary_table} WHERE namespace = :namespace")
        summaries = pd.read_sql(query, engine, params={"namespace": namespace})
    else:
        query = text(
            f"SELECT speaker_id, conversation_id, summary FROM {summary_table} "
            f"WHERE namespace = :namespace ORDER BY speaker_id, conversation_id"
        )
        rows = pd.read_sql(query, engine, params={"namespace": namespace})
        summaries = rows.groupby("speaker_id")["summary"].apply("\n\n".join).reset_index()

    summaries["speaker_id"] = summaries["speaker_id"].astype(str)
    return summaries.rename(columns={"speaker_id": user_field, "summary": "message"})


def build_combined_table(
    engine, full_df: pd.DataFrame, summary_df: pd.DataFrame, user_field: str, combined_table: str,
) -> None:
    """Stack full_df's and summary_df's rows into one corpus tagged with a
    binary `is_summary` column, restricted to persons present in both --
    otherwise the two sides wouldn't be the same underlying people/content,
    just differently represented.
    """
    matched = sorted(set(full_df[user_field]) & set(summary_df[user_field]))
    if not matched:
        raise SystemExit(f"no {user_field} is present in both the full-text and summary corpora")

    full_df = full_df[full_df[user_field].isin(matched)]
    summary_df = summary_df[summary_df[user_field].isin(matched)]

    combined = pd.concat(
        [
            full_df.assign(**{LABEL_FIELD: 0, GROUP_FIELD: lambda d: d[user_field] + "_full"}),
            summary_df.assign(**{LABEL_FIELD: 1, GROUP_FIELD: lambda d: d[user_field] + "_summary"}),
        ],
        ignore_index=True,
    )[[GROUP_FIELD, user_field, "message", LABEL_FIELD]]

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {combined_table}"))
        combined.to_sql(
            combined_table, conn, if_exists="append", index=False,
            dtype={GROUP_FIELD: String(255), user_field: String(255), "message": Text, LABEL_FIELD: Integer},
        )
        # to_sql's table defaults to the server's engine (MyISAM here), which caps
        # key length at 1000 bytes -- too short for a utf8mb4 VARCHAR(255) primary
        # key. InnoDB's default (DYNAMIC) row format supports keys up to 3072 bytes.
        conn.execute(text(f"ALTER TABLE {combined_table} ENGINE=InnoDB"))
        conn.execute(text(f"ALTER TABLE {combined_table} ADD PRIMARY KEY ({GROUP_FIELD})"))
    print(f"[{combined_table}: {len(matched)} matched {user_field}(s) x 2 rows "
          f"({len(full_df) - len(matched)} full-only, {len(summary_df) - len(matched)} summary-only dropped)]")


def extract_ngrams(
    database: str, message_table: str, feat_table: str, filtered_feat_table: str,
    set_p_occ: float, group_freq_thresh: int,
) -> None:
    """Step 1: 1-to-3gram feature table, occurrence-filtered (DLA tutorial's
    Step 1). --group_freq_thresh only applies to the *filter* call, which
    needs the per-document word-count table --add_ngrams just built.
    """
    command = [
        PYTHON, DLATK,
        "-d", database, "-t", message_table, "-c", GROUP_FIELD,
        "--add_ngrams", "-n", "1", "2", "3", "--combine_feat_tables", "1to3gram",
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)

    command = [
        PYTHON, DLATK,
        "-d", database, "-t", message_table, "-c", GROUP_FIELD,
        "-f", feat_table,
        "--feat_occ_filter", "--set_p_occ", str(set_p_occ),
        "--group_freq_thresh", str(group_freq_thresh),
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)
    print(f"[features -> {database}.{filtered_feat_table}]")


def correlate_wordclouds(
    database: str, message_table: str, filtered_feat_table: str,
    group_freq_thresh: int, output_dir: Path, output_name: str,
) -> None:
    """Step 2: correlate the filtered 1to3grams with `is_summary` and render
    positive/negative word clouds (DLA tutorial's Step 2). The message table
    doubles as its own outcome table since it already carries `is_summary`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_name
    command = [
        PYTHON, DLATK,
        "-d", database, "-t", message_table, "-c", GROUP_FIELD,
        "-f", filtered_feat_table,
        "--outcome_table", message_table, "--outcomes", LABEL_FIELD,
        "--group_freq_thresh", str(group_freq_thresh),
        "--output_name", str(stem),
        "--rmatrix", "--csv", "--sort",
        "--tagcloud", "--make_wordclouds",
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)
    print(f"[correlations -> {stem}.rMatrix.csv]")
    print(f"[wordclouds -> {stem}_tagcloud_wordclouds/]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default=DATABASE, help="MySQL database holding the raw message table; also where this script's working tables land (default: %(default)s)")
    parser.add_argument("--message_table", default=MESSAGE_TABLE, help="MySQL table holding the raw messages (default: %(default)s)")
    parser.add_argument("--conversation_field", default=CONVERSATION_FIELD, help="message_table's conversation column (default: %(default)s)")
    parser.add_argument("--user_field", default=USER_FIELD, help="message_table's user/person column (default: %(default)s)")
    parser.add_argument("--turn_field", default=TURN_FIELD, help="message_table's column giving each row's order within a conversation (default: %(default)s)")
    parser.add_argument("--memorydb", default=DATABASE, help="MySQL database the memory mechanism wrote summaries to (default: %(default)s)")
    parser.add_argument("--memory", choices=list(SUMMARY_TABLES), default="summary", help="which memory mechanism's table supplies the 'summary' side (default: %(default)s)")
    parser.add_argument("--summary_table", default=None, help="override the MySQL table read for --memory (default: derived, e.g. summaries_plain for --memory summary)")
    parser.add_argument("--namespace", required=True, help="namespace scoping which run's rows to read from the summary table (e.g. ds4ud_all_gpt4omini; see prediction.py's memory_namespace)")
    parser.add_argument("--set_p_occ", type=float, default=0.05, help="minimum fraction of documents an n-gram must occur in to survive --feat_occ_filter (default: %(default)s)")
    parser.add_argument("--group_freq_thresh", type=int, default=0, help="DLATK's minimum word count per document to be kept, applied during filtering/correlation (default: %(default)s)")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="where --correlate writes the rMatrix/tagcloud output (default: %(default)s)")
    parser.add_argument("--output_name", default=None, help="stem for --correlate's output files (default: ddla_<memory>_<namespace>)")
    parser.add_argument("--extract", action="store_true", help="Step 1: stack summary vs. full text and build the occurrence-filtered 1to3gram feature table")
    parser.add_argument("--correlate", action="store_true", help="Step 2: correlate features with is_summary and render word clouds")
    args = parser.parse_args()

    # No step flag means run both.
    if not (args.extract or args.correlate):
        args.extract = args.correlate = True

    summary_table = args.summary_table or SUMMARY_TABLES[args.memory]
    combined_table = f"ddla_{slug(args.memory)}_{slug(args.namespace)}"
    feat_table = f"feat$1to3gram${combined_table}${GROUP_FIELD}"
    filtered_feat_table = f"{feat_table}${str(args.set_p_occ).replace('.', '_')}"
    output_name = args.output_name or f"ddla_{slug(args.memory)}_{slug(args.namespace)}"

    engine = make_engine(args.database)
    memory_engine = engine if args.memorydb == args.database else make_engine(args.memorydb)

    if args.extract:
        full_df = read_full_text(engine, args.message_table, args.conversation_field, args.user_field, args.turn_field)
        summary_df = read_summary_text(memory_engine, args.memory, summary_table, args.namespace, args.user_field)
        build_combined_table(engine, full_df, summary_df, args.user_field, combined_table)
        extract_ngrams(args.database, combined_table, feat_table, filtered_feat_table, args.set_p_occ, args.group_freq_thresh)

    if args.correlate:
        correlate_wordclouds(args.database, combined_table, filtered_feat_table, args.group_freq_thresh, args.output_dir, output_name)
