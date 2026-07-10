"""Apply the pretrained OCEAN ridge pickle to `memllm.conversations_train`.

Follows https://dlatk.github.io/dlatk/tutorials/tut_pickle_apply.html

The counterpart to train_model.py, but the model is given rather than fitted, so
there is no outcome to regress on and no cross-validation. DLATK aligns extracted
features to the pickle's 704 1-to-3 grams by name and zero-fills the rest, hence
no --feat_occ_filter here.

--group picks the prediction unit: speaker_id (one per person, 71) or
conv_speaker (one per conversation-speaker, 974).

Three steps, all run when no step flag is passed:
  --prepare   build the zeroed dummy outcome table.
  --extract   1/2/3-gram features, combined into one 1to3gram table.
  --predict   apply the pickle -> feat$p_ridg_ocean$conversations_train$<group>.
"""

import argparse
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, text

### ---- Configuration ----

DATABASE = "memllm"

# `text`, not DLATK's default `message` (dlaConstants.DEF_MESSAGE_FIELD).
MESSAGE_TABLE = "conversations_train"
MESSAGE_FIELD = "text"
MESSAGEID_FIELD = "message_id"

GROUPS = ["speaker_id", "conv_speaker"]  # DLATK's -c; both are columns of MESSAGE_TABLE
GROUP_FREQ_THRESH = "0"  # keep every unit; a gft > 0 needs a 1gram word table

# The *pickle's* outcome keys, looked up in the loaded model, so a rename here
# silently predicts nothing. `neu_z` is neuroticism, not train_model.py's
# reverse-scored stability.
OUTCOMES = ["ope_z", "con_z", "ext_z", "agr_z", "neu_z"]
NGRAMS = ["1", "2", "3"]
FEATURE_NAME = "1to3gram"
PREDICTION_NAME = "ocean"  # DLATK prefixes it with "p_" + the pickled model's name[:4]

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = Path(__file__).resolve().parent
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
PICKLE_FILE = REPO_ROOT / "models" / "ocean_3dom.1to3g_ocean100.ridge10k.pickle"

# Zeroed table --predict_regression_to_feats reads its group list from.
OUTCOME_TABLE = "ocean_dummy"

### ---- Execution ----


def prepare(group):
    """Build the zeroed dummy outcome table keyed by `group`.

    --predict_regression_to_feats goes through the regression outcome path, so it
    needs an outcome table keyed like the `-c` group field (outcomeGetter.py:264,
    :332) even though the pickle supplies the model. Only the group list is read,
    never the values, hence zeros.
    """
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4")
    with engine.connect() as conn:
        
        conn.execute(text(f"DROP TABLE IF EXISTS {OUTCOME_TABLE}"))

        columns = ", ".join(f"{o} DOUBLE NOT NULL DEFAULT 0" for o in OUTCOMES)
        conn.execute(
            text(
                f"CREATE TABLE {OUTCOME_TABLE} ("
                f"  {group} VARCHAR(20) NOT NULL PRIMARY KEY, {columns}"
                f")"
            )
        )

        inserted = conn.execute(
            text(
                f"INSERT INTO {OUTCOME_TABLE} ({group}) "
                f"SELECT DISTINCT {group} FROM {MESSAGE_TABLE}"
            )
        )
        conn.commit()  # DDL autocommits, the INSERT does not

    print(f"[{OUTCOME_TABLE}: {inserted.rowcount} {group} units, zeroed]")


def extract(group):
    """1/2/3-grams per unit, combined into one 1to3gram table.

    The call chains two stages through args.feattable: extract each n
    (dlatkInterface.py:1038), combine into `1to3gram` (:1319).
    """
    command = [
        REPO_ROOT / ".venv" / "bin" / "python3", DLATK,
        "-d", DATABASE,
        "-t", MESSAGE_TABLE,
        "-c", group,
        "--message_field", MESSAGE_FIELD,
        "--messageid_field", MESSAGEID_FIELD,
        "--add_ngrams",
        "-n", *NGRAMS,
        "--combine_feat_tables", FEATURE_NAME,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here


def predict(group):
    """Apply the pickle, writing per-unit OCEAN scores to a feature table."""
    command = [
        REPO_ROOT / ".venv" / "bin" / "python3", DLATK,
        "-d", DATABASE,
        "-t", MESSAGE_TABLE,
        "-c", group,
        "--message_field", MESSAGE_FIELD,
        "--messageid_field", MESSAGEID_FIELD,
        "-f", f"feat${FEATURE_NAME}${MESSAGE_TABLE}${group}",
        "--outcome_table", OUTCOME_TABLE,
        "--outcomes", *OUTCOMES,
        "--group_freq_thresh", GROUP_FREQ_THRESH,
        "--load",
        "--picklefile", PICKLE_FILE,
        "--predict_regression_to_feats", PREDICTION_NAME,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    print(f"\n[predictions -> {DATABASE}.feat$p_ridg_{PREDICTION_NAME}${MESSAGE_TABLE}${group}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--group", choices=GROUPS, default="speaker_id",
        help="prediction unit for every step (default: %(default)s)",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Step 1: build the dummy outcome table",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help=f"Step 2: extract {FEATURE_NAME} features",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Step 3: apply the pickle, writing a prediction feature table",
    )
    args = parser.parse_args()

    # No step flag means run all three.
    if not (args.prepare or args.extract or args.predict):
        args.prepare = args.extract = args.predict = True

    if args.prepare:
        prepare(args.group)
    if args.extract:
        extract(args.group)
    if args.predict:
        predict(args.group)
