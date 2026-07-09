"""Apply the pretrained OCEAN ridge pickle to `memllm.conversations_train`.

Follows https://dlatk.github.io/dlatk/tutorials/tut_pickle_apply.html

One prediction per (conversation, speaker): DLATK's group field is `conv_speaker`,
materialised in the wassa2024.ipynb data prep and loaded with the message table.
The pickle holds five ridge models over 704 1-to-3 grams; DLATK aligns extracted
features to the model by name and zero-fills the rest.

Steps (all run when no flag is passed):
  1. --prepare   check message_id/conv_speaker, build the zeroed dummy outcome table.
  2. --extract   1/2/3-gram features, combined into one 1to3gram table.
  3. --predict   apply the pickle -> feat$p_ridg_ocean$conversations_train$conv_speaker.

dlatkInterface runs under a small shim (COMPAT_SHIM): it aliases numpy.linalg.linalg
(gone in numpy 2) and gives the 0.20-pickled Ridge a `positive` default so DLATK's
`str(regressor)` debug print works under a modern sklearn.
"""

import argparse
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, text

### ---- Model + data options ----

DATABASE = "memllm"
MESSAGE_TABLE = "conversations_train"
MESSAGE_FIELD = "text"
MESSAGEID_FIELD = "message_id"
GROUP_FIELD = "speaker_id"      # one prediction per (conversation, speaker)
GROUP_FREQ_THRESH = 0             # keep every group

PICKLE_FILE = Path(__file__).resolve().parents[2] / "models" / "ocean_3dom.1to3g_ocean100.ridge10k.pickle"
OUTCOMES = ["ope_z", "con_z", "ext_z", "agr_z", "neu_z"]  # must match the pickle's keys
NGRAMS = ["1", "2", "3"]
FEATURE_NAME = "1to3gram"

### ---- Plumbing ----

TUTORIAL_DIR = Path(__file__).resolve().parent
PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python3"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
OUTCOME_TABLE = "ocean_dummy"     # zeroed table --predict_regression_to_feats reads its groups from
FEATURE_TABLE = f"feat${FEATURE_NAME}${MESSAGE_TABLE}${GROUP_FIELD}"
PREDICTION_TABLE = f"feat$p_ridg_ocean${MESSAGE_TABLE}${GROUP_FIELD}"

# `python -c` puts cwd (not the script dir) on sys.path, so re-add it for `import dlatk`.
COMPAT_SHIM = """
import pathlib, runpy, sys
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
import numpy.linalg
sys.modules.setdefault("numpy.linalg.linalg", numpy.linalg)
import sklearn.linear_model
sklearn.linear_model.Ridge.positive = False
runpy.run_path(sys.argv[0], run_name="__main__")
"""

### ---- Execution ----


def run(args):
    """Invoke dlatkInterface.py under the compat shim, from the tutorial dir."""
    command = [PYTHON, "-c", COMPAT_SHIM, DLATK, *args]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)


def prepare():
    """Check the ids, then build the zeroed dummy outcome table keyed by conv_speaker."""
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4")
    with engine.begin() as conn:

        # A non-unique message id is silently deduped (dropped) by DLATK; fail loudly instead.
        rows, distinct_ids = conn.execute(
            text(f"SELECT COUNT(*), COUNT(DISTINCT {MESSAGEID_FIELD}) FROM {MESSAGE_TABLE}")
        ).one()
        if rows != distinct_ids:
            raise SystemExit(
                f"{MESSAGE_TABLE}.{MESSAGEID_FIELD} is not unique ({distinct_ids}/{rows} "
                f"distinct); DLATK would drop {rows - distinct_ids} messages."
            )

        # conv_speaker is built in wassa2024.ipynb and loaded with the table; just confirm it arrived.
        (has_column,) = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = :db "
                "AND table_name = :tbl AND column_name = :col"
            ),
            {"db": DATABASE, "tbl": MESSAGE_TABLE, "col": GROUP_FIELD},
        ).one()
        if not has_column:
            raise SystemExit(
                f"{MESSAGE_TABLE}.{GROUP_FIELD} is missing; re-run the wassa2024.ipynb data "
                f"prep and reload the table."
            )

        # --predict_regression_to_feats reads its group list from an outcome table -- a zeroed one.
        print(f"[rebuilding dummy outcome table {OUTCOME_TABLE}]")
        columns = ", ".join(f"{o} DOUBLE NOT NULL DEFAULT 0" for o in OUTCOMES)
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {OUTCOME_TABLE}")
        conn.exec_driver_sql(
            f"CREATE TABLE {OUTCOME_TABLE} ("
            f"  {GROUP_FIELD} VARCHAR(16) NOT NULL PRIMARY KEY, {columns}"
            f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        inserted = conn.execute(
            text(
                f"INSERT INTO {OUTCOME_TABLE} ({GROUP_FIELD}) "
                f"SELECT DISTINCT {GROUP_FIELD} FROM {MESSAGE_TABLE}"
            )
        )
        print(f"  {inserted.rowcount} (conversation, speaker) units")
    engine.dispose()


def extract():
    """1/2/3-grams per (conversation, speaker), combined into one 1to3gram table."""
    run(
        [
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", GROUP_FIELD,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "--add_ngrams",
            "-n", *NGRAMS,
            "--combine_feat_tables", FEATURE_NAME,
        ]
    )


def predict():
    """Apply the pickle, writing per (conversation, speaker) OCEAN scores to a feature table."""
    run(
        [
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", GROUP_FIELD,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "--group_freq_thresh", str(GROUP_FREQ_THRESH),
            "-f", FEATURE_TABLE,
            "--outcome_table", OUTCOME_TABLE,
            "--outcomes", *OUTCOMES,
            "--predict_regression_to_feats", "ocean",
            "--load",
            "--picklefile", PICKLE_FILE,
        ]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepare", action="store_true", help="Step 1: check ids, build the dummy outcome table")
    parser.add_argument("--extract", action="store_true", help="Step 2: extract 1to3gram features")
    parser.add_argument("--predict", action="store_true", help=f"Step 3: apply the pickle, writing {PREDICTION_TABLE}")
    args = parser.parse_args()

    # No flags means run the whole pipeline.
    if not (args.prepare or args.extract or args.predict):
        args.prepare = args.extract = args.predict = True

    if args.prepare:
        prepare()
    if args.extract:
        extract()
    if args.predict:
        predict()
        print(f"\n[predictions written to {DATABASE}.{PREDICTION_TABLE}]")
