import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import Double, MetaData, Table, create_engine, text
from sqlalchemy.dialects.mysql import TINYINT

### ---- Configuration ----

MESSAGE_FIELD = "message"
MESSAGEID_FIELD = "message_id"
FOLD_FIELD = "fold"

# Per-corpus defaults, selected by --dataset; any field can still be overridden individually.
DATASETS = {
    "candor": dict(
        database="candor",
        message_table="msgsc",  # already uses DLATK's default `message` field, keyed by `message_id`
        outcome_table="surveys",
        outcome_key="user_id",  # OUTCOME_TABLE's person column; renamed to --group before use
        group="speaker",
        nfolds=10,
        outcomes=[
            "my_open",
            "my_conscientious",
            "my_extraversion",
            "my_agreeable",
            "my_neurotic",  # CANDOR reports neuroticism directly (not WASSA's reverse-scored stability)
        ],
    ),
    "ds4ud": dict(
        database="ssubrahmanya",
        message_table="msg_essays_v9v11",
        outcome_table="outcomes_v9v11_person",
        outcome_key=None,  # OUTCOME_TABLE already keys on the group column
        group="person_id",
        nfolds=10,
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
        message_table="msg_essays",  # copied from states_and_traits
        outcome_table="outcomes_user_2m2w",  # copied from states_and_traits
        outcome_key=None,  # both tables key on `user_id`
        group="user_id",
        nfolds=10,
        outcomes=[
            "avg_from_wave_openness_score",
            "avg_from_wave_conscientious_score",
            "avg_from_wave_extravert_score",
            "avg_from_wave_agreeable_score",
            "avg_from_wave_neurotic_score",
        ],
    ),
}

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

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"

### ---- Execution ----


def prepare_outcome_table(group, database, message_table, outcome_table, outcome_key, outcomes, nfolds, fold_field):
    """Rebuild `big5_<group>`: one Big-5 vector per group, plus a fold label.

    DLATK cannot regress on OUTCOME_TABLE directly: it needs an outcome table keyed on a column
    named like the `-c` group field (outcomeGetter.py:264, :332). OUTCOME_TABLE may key on a
    differently-named person column (--outcome_key, e.g. CANDOR's `user_id`), so that column is
    renamed to `group` first.

    Averaging by `group` then handles both shapes uniformly: DS4UD already has one row per
    person, so the mean is a no-op; CANDOR surveys per conversation, so a speaker's several
    slightly different OCEAN vectors collapse into one. Units with no usable label (all-NaN
    outcomes) are dropped, not treated as a fatal integrity error.

    Restricting to speakers who appear in MESSAGE_TABLE matters even after averaging: a
    surveyed person who never speaks would otherwise carry a real label with no language
    signal at all, which DLATK's --train_regression path pads to an all-zero feature vector
    rather than dropping (regressionPredictor.py:616's XGroups.union(groups) reintroduces
    exactly these outcome-only groups; alignDictsAsXy then zero-fills their row).

    The group column is typed from MESSAGE_TABLE; pandas infers TEXT, too wide for a primary
    key and unindexed for DLATK's join to the feature table's group_id.
    """
    table = f"big5_{group}"
    key = outcome_key or group
    engine = create_engine(
        f"mysql://ssubrahmanya@localhost/{database}?charset=utf8mb4&read_default_file=~/.my.cnf"
    )

    speaking = pd.read_sql(f"SELECT DISTINCT {group} FROM {message_table}", engine)
    raw = pd.read_sql(f"SELECT {key}, {', '.join(outcomes)} FROM {outcome_table}", engine)
    raw = raw.rename(columns={key: group})
    # Match types across the two tables' person column (e.g. DS4UD's outcome table is INT
    # where MESSAGE_TABLE's is VARCHAR); harmless when both are already the same type.
    raw[group] = raw[group].astype(str)
    speaking[group] = speaking[group].astype(str)

    labelled = raw.groupby(group, as_index=False)[outcomes].mean()
    labelled = labelled.dropna(subset=outcomes)
    labelled = labelled[labelled[group].isin(speaking[group])]

    labelled[fold_field] = pd.factorize(labelled[group], sort=True)[0] % nfolds
    outcomes_out = labelled[[group, fold_field, *outcomes]]

    dropped = len(speaking) - len(outcomes_out)

    key_type = Table(message_table, MetaData(), autoload_with=engine).c[group].type
    # to_sql declares neither PRIMARY KEY nor NOT NULL, so restate both.
    tighten = ", ".join(
        [f"MODIFY {fold_field} TINYINT UNSIGNED NOT NULL"]
        + [f"MODIFY {o} DOUBLE NOT NULL" for o in outcomes]
    )
    with engine.begin() as conn:
        # "replace" drops and recreates, so a failure above leaves the previous table
        # standing; the ALTER shares this transaction with it.
        outcomes_out.to_sql(
            table, conn, if_exists="replace", index=False,
            dtype={group: key_type, fold_field: TINYINT(unsigned=True),
                   **{o: Double for o in outcomes}},
        )
        conn.execute(text(f"ALTER TABLE {table} ADD PRIMARY KEY ({group}), {tighten}"))
    engine.dispose()
    print(
        f"[{table}: {len(outcomes_out)} labelled of {len(speaking)} speaking "
        f"({dropped} dropped for missing outcomes), {nfolds} folds]"
    )


def extract(group, features, database, message_table, message_field, messageid_field, group_freq_thresh):
    """Build the feature table named by `features`.

    The n-gram call chains three stages through args.feattable: extract each n, combine into
    `1to3gram`, filter (dlatkInterface.py:1038, :1319, :1343). --set_p_occ is a *proportion of
    groups*, so 0.05 keeps n-grams used by at least ~5% of groups.

    `group_freq_thresh` is shared with train() and predict() (all three come from the same
    parsed --group_freq_thresh) so every feature space scores over the same rows.
    """
    if features == "ngram":
        command = [
            PYTHON, DLATK,
            "-d", database,
            "-t", message_table,
            "-c", group,
            "--message_field", message_field,
            "--messageid_field", messageid_field,
            "--add_ngrams",
            "-n", "1", "2", "3",
            "--combine_feat_tables", "1to3gram",
            "--feat_occ_filter",
            "--set_p_occ", "0.05",
            "--group_freq_thresh", group_freq_thresh,
        ]
    else:
        command = [
            PYTHON, DLATK,
            "-d", database,
            "-t", message_table,
            "-c", group,
            "--message_field", message_field,
            "--messageid_field", messageid_field,
            "--add_emb_feat",
            "--embedding_model", "roberta-base",
            "--embedding_msg_aggregation", "mean",  # over messages
            "--embedding_word_aggregation", "mean",  # over word pieces
            "--embedding_layer_aggregation", "concatenate",
            "--embedding_layers", "11",  # second-to-last of 12
        ]

      # dlatk/ lives here


def train(
    group, features, database, message_table, output_dir, outcomes,
    message_field, messageid_field, fold_field, group_freq_thresh, picklefile=None,
):
    """Cross-validated fit; with `picklefile`, also a final model over all units.

    Two runs, not one: --output_name persists the metrics, but with --train_regression it also
    sets saveFeatures (dlatkInterface.py:1889), dumping the dense design matrix per outcome
    (regressionPredictor.py:658).

    --nfold_regression saves no model -- testControlCombos never assigns self.regressionModels.
    No --folds: --fold_column already fixes the count.
    """
    stem = output_dir / FEATURES[features]["output_stem"].format(group=group)
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON, DLATK,
        "-d", database,
        "-t", message_table,
        "-c", group,
        "--message_field", message_field,
        "--messageid_field", messageid_field,
        "-f", FEATURES[features]["table"].format(corptable=message_table, group=group),
        "--outcome_table", f"big5_{group}",
        "--outcomes", *outcomes,
        "--group_freq_thresh", group_freq_thresh,
        "--model", FEATURES[features]["model"],
        "--nfold_regression",
        "--fold_column", fold_field,
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
            "-d", database,
            "-t", message_table,
            "-c", group,
            "--message_field", message_field,
            "--messageid_field", messageid_field,
            "-f", FEATURES[features]["table"].format(corptable=message_table, group=group),
            "--outcome_table", f"big5_{group}",
            "--outcomes", *outcomes,
            "--group_freq_thresh", group_freq_thresh,
            "--model", FEATURES[features]["model"],
            "--train_regression", "--save", "--picklefile", picklefile,
        ]
        subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
        print(f"[model -> {picklefile}]")


def predict(group, features, database, message_table, outcomes, message_field, messageid_field, group_freq_thresh, picklefile=None):
    """Write predicted Big-5 per unit to a feature table.

    Without `picklefile`, fits first: dlatkInterface runs load, train, predict_to_feats, save
    in that order against one RegressionPredictor (:1883-1957). Omitting --output_name stops
    --train_regression from also dumping the design matrix.
    """
    fit = ["--load", "--picklefile", picklefile] if picklefile else ["--train_regression"]
    name = FEATURES[features]["prediction_name"]
    command = [
        PYTHON, DLATK,
        "-d", database,
        "-t", message_table,
        "-c", group,
        "--message_field", message_field,
        "--messageid_field", messageid_field,
        "-f", FEATURES[features]["table"].format(corptable=message_table, group=group),
        "--outcome_table", f"big5_{group}",
        "--outcomes", *outcomes,
        "--group_freq_thresh", group_freq_thresh,
        "--model", FEATURES[features]["model"],
        *fit,
        "--predict_regression_to_feats", name,
    ]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)  # dlatk/ lives here
    table = f"feat$p_{FEATURES[features]['model'][:4]}_{name}${message_table}${group}"
    print(f"\n[predictions -> {database}.{table}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), required=True,
        help="corpus preset for --database/--message_table/--outcome_table/--outcome_key/"
             "--group/--nfolds/--outcomes (each still individually overridable)",
    )
    parser.add_argument(
        "--database", default=None,
        help="MySQL database holding the message/outcome tables (default: --dataset preset)",
    )
    parser.add_argument(
        "--message_table", default=None,
        help="message table, shaped like DLATK's default (default: --dataset preset)",
    )
    parser.add_argument(
        "--outcome_table", default=None,
        help="table holding one Big-5 vector per --outcome_key (default: --dataset preset)",
    )
    parser.add_argument(
        "--outcome_key", default=None,
        help="OUTCOME_TABLE's person column, if named differently from --group "
             "(default: --dataset preset)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="where --train writes metrics/predicted-data CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--group", default=None,
        help="training unit for every step (default: --dataset preset)",
    )
    parser.add_argument(
        "--nfolds", type=int, default=None,
        help="number of cross-validation folds (default: --dataset preset)",
    )
    parser.add_argument(
        "--message_field", default=MESSAGE_FIELD,
        help="message table's text column (default: %(default)s)",
    )
    parser.add_argument(
        "--messageid_field", default=MESSAGEID_FIELD,
        help="message table's message-id column (default: %(default)s)",
    )
    parser.add_argument(
        "--fold_field", default=FOLD_FIELD,
        help="column name for the fold label written into big5_<group> (default: %(default)s)",
    )
    parser.add_argument(
        "--group_freq_thresh", type=int, default=100,
        help="DLATK's minimum language amount (word count) a unit needs to be kept, shared by "
             "--extract/--train/--predict (default: %(default)s)",
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
        help="Step 2: cross-validated fit, writing metrics to --output_dir; "
             "saves a model too if --picklefile is given",
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

    preset = DATASETS[args.dataset]
    args.database = args.database or preset["database"]
    args.message_table = args.message_table or preset["message_table"]
    args.outcome_table = args.outcome_table or preset["outcome_table"]
    args.outcome_key = args.outcome_key or preset["outcome_key"]
    args.group = args.group or preset["group"]
    args.nfolds = args.nfolds or preset["nfolds"]
    outcomes = preset["outcomes"]

    # No step flag means run all three.
    if not (args.extract or args.train or args.predict):
        args.extract = args.train = args.predict = True

    prepare_outcome_table(
        args.group, args.database, args.message_table, args.outcome_table,
        args.outcome_key, outcomes, args.nfolds, args.fold_field,
    )

    if args.extract:
        extract(
            args.group, args.features, args.database, args.message_table,
            args.message_field, args.messageid_field, args.group_freq_thresh,
        )
    if args.train:
        train(
            args.group, args.features, args.database, args.message_table,
            args.output_dir, outcomes, args.message_field, args.messageid_field,
            args.fold_field, args.group_freq_thresh, args.picklefile,
        )
    if args.predict:
        predict(
            args.group, args.features, args.database, args.message_table, outcomes,
            args.message_field, args.messageid_field, args.group_freq_thresh, args.picklefile,
        )
