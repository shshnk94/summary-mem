"""Predict the Big-5 inventory from `candor.msgsc` language.

Follows https://dlatk.github.io/dlatk/tutorials/tut_pred.html.

DLATK runs directly against the CANDOR corpus -- `msgsc` is already a message table in the
shape DLATK wants (a `message` field keyed by `message_id`), and `surveys` already holds the
self-reported OCEAN. Nothing is copied or renamed; only the outcome table below is derived.

--features  ngram (occurrence-filtered 1-to-3 grams, ridgecv) or roberta-base
            (mean-pooled layer-11 embeddings, ridgecv).
--group     speaker -- one unit per person (~1,431 labelled of 1,456). Big-5 is a stable
            per-person trait, so the person is the natural unit; folding by speaker also keeps
            the same person out of train and test, which would otherwise let the model measure
            speaker re-identification (n-grams exploit that harder than embeddings, a reused
            rare trigram being a near-unique fingerprint).

Two CANDOR specifics handled in prepare_outcome_table (vs. the WASSA tutorial):
  * CANDOR surveys each participant per conversation, so a person can carry several slightly
    different OCEAN vectors -- we average them into one label per speaker.
  * ~25 participants have no usable survey -- their units are dropped, not treated as errors.

The outcome table is rebuilt on every invocation. The three steps below all run when no step
flag is passed:
  --extract   build the feature table named by --features.
  --train     cross-validated fit to output/; saves a model if --picklefile.
  --predict   predicted Big-5 to a feature table, applying --picklefile if given,
              otherwise fitting first.
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import Double, MetaData, Table, create_engine, text
from sqlalchemy.dialects.mysql import TINYINT

### ---- Configuration ----

DATABASE = "candor"

# `msgsc` already uses DLATK's default `message` field, keyed by `message_id`.
MESSAGE_TABLE = "msgsc"
MESSAGE_FIELD = "message"
MESSAGEID_FIELD = "message_id"

OUTCOME_TABLE = "surveys"
SPEAKER_FIELD = "speaker"  # `user_id` in OUTCOME_TABLE
FOLD_FIELD = "fold"
OUTCOMES = [
    "my_open",
    "my_conscientious",
    "my_extraversion",
    "my_agreeable",
    "my_neurotic",  # CANDOR reports neuroticism directly (not WASSA's reverse-scored stability)
]

GROUPS = ["speaker"]  # DLATK's -c; a column of MESSAGE_TABLE
NFOLDS = 10
GROUP_FREQ_THRESH = "0"  # keep every unit; a gft > 0 needs a 1gram word table

# `table` is the name DLATK derives, so it must track extract()'s flags:
# roberta-base + mean + L11 + concatenate -> roberta_ba_meL11con
# (featureExtractor.py:1328); n-grams get the --set_p_occ suffix, '.' -> '_'
# (featureRefiner.createTableWithRemovedFeats:286). Wrong name and extraction
# writes one table while train/predict read another.
FEATURES = {
    "roberta-base": {
        "table": "feat$roberta_ba_meL11con${corptable}${group}",
        # RidgeCV picks alpha by LOO within each training fold -- a fixed alpha is
        # arbitrary. Grid: [1000, 0.1, 1, 10, 100, 1e4, 1e5].
        "model": "ridgecv",  # a key of RegressionPredictor.cvParams
        "prediction_name": "big5",  # DLATK prefixes it with "p_" + model[:4]
        "output_stem": "big5_{group}_roberta-base_L11_ridgecv",
    },
    "ngram": {
        "table": "feat$1to3gram${corptable}${group}$0_05",
        "model": "ridgecv",  # sparse matrix; the right alpha is not knowable up front
        "prediction_name": "big5ng",
        "output_stem": "big5_{group}_1to3gram_p0_05_ridgecv",
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"

### ---- Execution ----


def prepare_outcome_table(group):
    """Rebuild `big5_<group>`: one averaged Big-5 vector per speaker, plus a fold label.

    DLATK cannot regress on `surveys` directly: it needs an outcome table keyed on a column
    named like the `-c` group field (outcomeGetter.py:264, :332), and `surveys` keys on
    (conversation_id, user_id). Joining through MESSAGE_TABLE also drops surveyed people who
    never speak -- otherwise an all-zero feature vector carrying a real label
    (featureGetter.py:336).

    Unlike WASSA (one label per person), CANDOR surveys per conversation, so a speaker may
    carry several OCEAN vectors -- we average them into one label. Units with no usable label
    (all-NaN outcomes) are dropped, not treated as a fatal integrity error.

    The group column is typed from MESSAGE_TABLE; pandas infers TEXT, too wide for a primary
    key and unindexed for DLATK's join to the feature table's group_id.
    """
    table = f"big5_{group}"
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4")

    messages = pd.read_sql(
        f"SELECT DISTINCT conversation_id, {SPEAKER_FIELD} FROM {MESSAGE_TABLE}", engine
    )
    surveys = pd.read_sql(
        f"SELECT conversation_id, user_id, {', '.join(OUTCOMES)} FROM {OUTCOME_TABLE}", engine
    )

    surveyed = messages.merge(
        surveys,
        left_on=["conversation_id", SPEAKER_FIELD],
        right_on=["conversation_id", "user_id"],
    )

    # Average a speaker's (possibly several) per-conversation survey vectors into one label.
    labelled = surveyed.groupby(group, as_index=False)[OUTCOMES].mean()

    # Drop speakers whose label is entirely missing (mean over all-NaN survey rows is NaN).
    labelled = labelled.dropna(subset=OUTCOMES)

    # Fold by speaker -- at this grouping level the unit *is* the person, so a person cannot
    # straddle folds.
    labelled[FOLD_FIELD] = pd.factorize(labelled[group], sort=True)[0] % NFOLDS
    outcomes = labelled[[group, FOLD_FIELD, *OUTCOMES]]

    speaks = messages[group].drop_duplicates()
    dropped = len(speaks) - len(outcomes)

    key_type = Table(MESSAGE_TABLE, MetaData(), autoload_with=engine).c[group].type
    # to_sql declares neither PRIMARY KEY nor NOT NULL, so restate both.
    tighten = ", ".join(
        [f"MODIFY {FOLD_FIELD} TINYINT UNSIGNED NOT NULL"]
        + [f"MODIFY {o} DOUBLE NOT NULL" for o in OUTCOMES]
    )
    with engine.begin() as conn:
        # "replace" drops and recreates, so a failure above leaves the previous table
        # standing; the ALTER shares this transaction with it.
        outcomes.to_sql(
            table, conn, if_exists="replace", index=False,
            dtype={group: key_type, FOLD_FIELD: TINYINT(unsigned=True),
                   **{o: Double for o in OUTCOMES}},
        )
        conn.execute(text(f"ALTER TABLE {table} ADD PRIMARY KEY ({group}), {tighten}"))
    engine.dispose()
    print(
        f"[{table}: {len(outcomes)} labelled speakers of {len(speaks)} speaking "
        f"({dropped} dropped for missing outcomes), {NFOLDS} folds]"
    )


def extract(group, features):
    """Build the feature table named by `features`.

    The n-gram call chains three stages through args.feattable: extract each n, combine into
    `1to3gram`, filter (dlatkInterface.py:1038, :1319, :1343). --set_p_occ is a *proportion of
    groups*, so 0.05 keeps n-grams used by at least ~5% of speakers.

    GROUP_FREQ_THRESH stays 0 so both feature spaces score over the same rows.
    """
    if features == "ngram":
        command = [
            PYTHON, DLATK,
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", group,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "--add_ngrams",
            "-n", "1", "2", "3",
            "--combine_feat_tables", "1to3gram",
            "--feat_occ_filter",
            "--set_p_occ", "0.05",
            "--group_freq_thresh", GROUP_FREQ_THRESH,
        ]
    else:
        command = [
            PYTHON, DLATK,
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", group,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "--add_emb_feat",
            "--embedding_model", "roberta-base",
            "--embedding_msg_aggregation", "mean",  # over turns
            "--embedding_word_aggregation", "mean",  # over word pieces
            "--embedding_layer_aggregation", "concatenate",
            "--embedding_layers", "11",  # second-to-last of 12
        ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here


def train(group, features, picklefile=None):
    """Cross-validated fit; with `picklefile`, also a final model over all units.

    Two runs, not one: --output_name persists the metrics, but with --train_regression it also
    sets saveFeatures (dlatkInterface.py:1889), dumping the dense design matrix per outcome
    (regressionPredictor.py:658).

    --nfold_regression saves no model -- testControlCombos never assigns self.regressionModels.
    No --folds: --fold_column already fixes the count.
    """
    stem = OUTPUT_DIR / FEATURES[features]["output_stem"].format(group=group)
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON, DLATK,
        "-d", DATABASE,
        "-t", MESSAGE_TABLE,
        "-c", group,
        "--message_field", MESSAGE_FIELD,
        "--messageid_field", MESSAGEID_FIELD,
        "-f", FEATURES[features]["table"].format(corptable=MESSAGE_TABLE, group=group),
        "--outcome_table", f"big5_{group}",
        "--outcomes", *OUTCOMES,
        "--group_freq_thresh", GROUP_FREQ_THRESH,
        "--model", FEATURES[features]["model"],
        "--nfold_regression",
        "--fold_column", FOLD_FIELD,
        # Both suffix --output_name (dlatkInterface.py:1916): .accuracy_data.csv
        # (r, R2, mse, mae, N, num_features per outcome) and .predicted_data.csv.
        "--csv",
        "--pred_csv",
        "--output_name", stem,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    print(f"[metrics -> {stem}.accuracy_data.csv]")
    print(f"[fold predictions -> {stem}.predicted_data.csv]")

    if picklefile:
        Path(picklefile).parent.mkdir(parents=True, exist_ok=True)
        command = [
            PYTHON, DLATK,
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", group,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "-f", FEATURES[features]["table"].format(corptable=MESSAGE_TABLE, group=group),
            "--outcome_table", f"big5_{group}",
            "--outcomes", *OUTCOMES,
            "--group_freq_thresh", GROUP_FREQ_THRESH,
            "--model", FEATURES[features]["model"],
            "--train_regression", "--save", "--picklefile", picklefile,
        ]
        subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
        print(f"[model -> {picklefile}]")


def predict(group, features, picklefile=None):
    """Write predicted Big-5 per unit to a feature table.

    Without `picklefile`, fits first: dlatkInterface runs load, train, predict_to_feats, save
    in that order against one RegressionPredictor (:1883-1957). Omitting --output_name stops
    --train_regression from also dumping the design matrix.
    """
    fit = ["--load", "--picklefile", picklefile] if picklefile else ["--train_regression"]
    name = FEATURES[features]["prediction_name"]
    command = [
        PYTHON, DLATK,
        "-d", DATABASE,
        "-t", MESSAGE_TABLE,
        "-c", group,
        "--message_field", MESSAGE_FIELD,
        "--messageid_field", MESSAGEID_FIELD,
        "-f", FEATURES[features]["table"].format(corptable=MESSAGE_TABLE, group=group),
        "--outcome_table", f"big5_{group}",
        "--outcomes", *OUTCOMES,
        "--group_freq_thresh", GROUP_FREQ_THRESH,
        "--model", FEATURES[features]["model"],
        *fit, 
        "--predict_regression_to_feats", name,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    table = f"feat$p_{FEATURES[features]['model'][:4]}_{name}${MESSAGE_TABLE}${group}"
    print(f"\n[predictions -> {DATABASE}.{table}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--group", choices=GROUPS, default="speaker",
        help="training unit for every step (default: %(default)s)",
    )
    parser.add_argument(
        "--features", choices=sorted(FEATURES), default="roberta-base",
        help="feature space for every step (default: %(default)s)",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Step 1: build the feature table named by --features",
    )
    parser.add_argument(
        "--train", action="store_true",
        help=f"Step 2: {NFOLDS}-fold cross-validated fit, writing metrics to output/; "
             f"saves a model too if --picklefile is given",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Step 3: write predicted Big-5 to a feature table, using --picklefile "
             "if given, otherwise training first",
    )
    parser.add_argument(
        "--picklefile", type=Path, default=None,
        help="--train saves the fitted model here; --predict loads it from here",
    )
    args = parser.parse_args()

    # No step flag means run all three.
    if not (args.extract or args.train or args.predict):
        args.extract = args.train = args.predict = True

    prepare_outcome_table(args.group)

    if args.extract:
        extract(args.group, args.features)
    if args.train:
        train(args.group, args.features, args.picklefile)
    if args.predict:
        predict(args.group, args.features, args.picklefile)
