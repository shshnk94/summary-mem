"""Predict the Big-5 inventory from `memllm.conversations_train` language.

Follows https://dlatk.github.io/dlatk/tutorials/tut_pred.html.

--features  ngram (occurrence-filtered 1-to-3 grams, ridgecv) or roberta-base
            (mean-pooled layer-11 embeddings, ridgecv).
--group     speaker_id (71 people) or conv_speaker (974 pairs -- more rows, but a
            person recurs across their conversations).

Big-5 is a stable per-person trait, so at conv_speaker level random k-fold CV
would put the same person in train and test and partly measure speaker
re-identification -- which n-grams exploit harder than embeddings, a reused rare
trigram being a near-unique fingerprint. prepare_outcome_table() therefore folds
by *speaker* at both levels, and --nfold_regression honours that via
--fold_column.

The outcome table is rebuilt on every invocation. The three steps below all run
when no step flag is passed:
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

DATABASE = "memllm"

# `text`, not DLATK's default `message` (dlaConstants.DEF_MESSAGE_FIELD).
MESSAGE_TABLE = "conversations_train"
MESSAGE_FIELD = "text"
MESSAGEID_FIELD = "message_id"

OUTCOME_TABLE = "surveys_train"
SPEAKER_FIELD = "speaker_id"  # `person_id` in OUTCOME_TABLE
FOLD_FIELD = "fold"
OUTCOMES = [
    "personality_openness",
    "personality_conscientiousness",
    "personality_extraversion",
    "personality_agreeableness",
    "personality_stability",  # reverse-scored neuroticism
]

GROUPS = ["speaker_id", "conv_speaker"]  # DLATK's -c; both are columns of MESSAGE_TABLE
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
        # arbitrary at n=71, p=768. Grid: [1000, 0.1, 1, 10, 100, 1e4, 1e5].
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
    """Rebuild `big5_<group>`: Big-5 per unit, plus a speaker-disjoint fold label.

    DLATK cannot regress on surveys_train directly: it needs an outcome table
    keyed on a column named like the `-c` group field (outcomeGetter.py:264,
    :332), and surveys_train keys on `id`.

    Joining through MESSAGE_TABLE lets one schema serve both group fields, and
    drops surveyed pairs that never speak -- otherwise an all-zero feature vector
    carrying a real label (featureGetter.py:336). pd.merge rather than SQL, so
    `messages` stays in hand for the unsurveyed check below.

    Ranking the *speaker* puts all of one person's rows in one fold at either
    grouping level. Beware: testControlCombos overrides --folds with the number
    of distinct labels (regressionPredictor.py:815-822).

    The group column is typed from MESSAGE_TABLE; pandas infers TEXT, too wide
    for a primary key and unindexed for DLATK's join to the feature table's
    group_id.
    """
    table = f"big5_{group}"
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4")

    # dict.fromkeys dedups in order: at speaker_id level `group` is SPEAKER_FIELD,
    # and naming a column twice would merge into duplicate columns.
    unit_columns = list(dict.fromkeys(["conversation_id", SPEAKER_FIELD, group]))
    keep = list(dict.fromkeys([group, SPEAKER_FIELD, *OUTCOMES]))

    messages = pd.read_sql(f"SELECT DISTINCT {', '.join(unit_columns)} FROM {MESSAGE_TABLE}", engine)
    surveys = pd.read_sql(
        f"SELECT conversation_id, person_id, {', '.join(OUTCOMES)} FROM {OUTCOME_TABLE}", engine
    )

    # Collapses a unit's conversations to one row -- unless they carry more than
    # one Big-5 vector, which the duplicate check below catches.
    surveyed = messages.merge(
        surveys,
        left_on=["conversation_id", SPEAKER_FIELD],
        right_on=["conversation_id", "person_id"],
    )[keep].drop_duplicates()

    dupes = surveyed[group][surveyed[group].duplicated()].unique()
    if len(dupes):
        raise SystemExit(
            f"{len(dupes)} {group} units carry more than one Big-5 vector: {list(dupes[:5])}"
        )

    # A missing or partial outcome row silently shrinks the training set.
    speaks = messages[group].drop_duplicates()
    missing = speaks[~speaks.isin(surveyed[group])]
    if len(missing):
        raise SystemExit(
            f"{len(missing)} of {len(speaks)} {group} units speak but have no survey "
            f"row: {list(missing[:5])}"
        )
    blank = surveyed[group][surveyed[OUTCOMES].isna().any(axis=1)]
    if len(blank):
        raise SystemExit(f"{len(blank)} {group} units have a null outcome: {list(blank[:5])}")

    surveyed[FOLD_FIELD] = pd.factorize(surveyed[SPEAKER_FIELD], sort=True)[0] % NFOLDS
    outcomes = surveyed[[group, FOLD_FIELD, *OUTCOMES]]

    key_type = Table(MESSAGE_TABLE, MetaData(), autoload_with=engine).c[group].type
    # to_sql declares neither PRIMARY KEY nor NOT NULL, so restate both.
    tighten = ", ".join(
        [f"MODIFY {FOLD_FIELD} TINYINT UNSIGNED NOT NULL"]
        + [f"MODIFY {o} DOUBLE NOT NULL" for o in OUTCOMES]
    )
    with engine.begin() as conn:
        # "replace" drops and recreates, so a failure above leaves the previous
        # table standing; the ALTER shares this transaction with it.
        outcomes.to_sql(
            table, conn, if_exists="replace", index=False,
            dtype={group: key_type, FOLD_FIELD: TINYINT(unsigned=True),
                   **{o: Double for o in OUTCOMES}},
        )
        conn.execute(text(f"ALTER TABLE {table} ADD PRIMARY KEY ({group}), {tighten}"))
    engine.dispose()
    print(f"[{table}: {len(outcomes)} units, {NFOLDS} speaker-disjoint folds]")


def extract(group, features):
    """Build the feature table named by `features`.

    The n-gram call chains three stages through args.feattable: extract each n,
    combine into `1to3gram`, filter (dlatkInterface.py:1038, :1319, :1343).
    --set_p_occ is a *proportion of groups*, so 0.05 means two different things:
    four speakers at speaker_id level, forty-nine conversations at conv_speaker.

    GROUP_FREQ_THRESH stays 0 so both spaces score over the same rows; DLATK's
    default of 1000 would empty this corpus (233 words per unit).
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

    Two runs, not one: --output_name persists the metrics, but with
    --train_regression it also sets saveFeatures (dlatkInterface.py:1889),
    dumping the dense design matrix per outcome (regressionPredictor.py:658).

    --nfold_regression saves no model -- testControlCombos never assigns
    self.regressionModels. No --folds: --fold_column already fixes the count.
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

    Without `picklefile`, fits first: dlatkInterface runs load, train,
    predict_to_feats, save in that order against one RegressionPredictor
    (:1883-1957). Omitting --output_name stops --train_regression from also
    dumping the design matrix.
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
        *fit, "--predict_regression_to_feats", name,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    table = f"feat$p_{FEATURES[features]['model'][:4]}_{name}${MESSAGE_TABLE}${group}"
    print(f"\n[predictions -> {DATABASE}.{table}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--group", choices=GROUPS, default="speaker_id",
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
