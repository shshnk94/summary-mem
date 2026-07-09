"""Predict the Big-5 inventory from `memllm.conversations_train` language.

Ridge regression over mean-pooled roberta-large embeddings, one training unit
per (conversation, speaker). Every conversation has exactly 2 speakers, so the
487 conversations yield 974 units -- a person who talks in several conversations
contributes one unit per conversation, each modelled independently.

The five Big-5 targets live in `memllm.surveys_train` as personality_openness,
_conscientiousness, _extraversion, _agreeableness and _stability, one row per
(conversation_id, person_id) -- exactly the granularity we model at. DLATK still
cannot read that table directly: OutcomeGetter selects `[correl_field,
outcomeField]` from it (outcomeGetter.py:264) and calls checkIndices(...,
primary=True) (:332), so the outcome table needs a *primary key* column named
exactly like the `-c` group field. surveys_train keys on `id` instead.

DLATK's group field must also be a real column of the message table, and there
is no single column identifying a (conversation, speaker). So --prepare
  * adds a stored generated column `conv_speaker` = '<conversation_id>_<speaker_id>'
    to conversations_train, indexed, and
  * builds `big5_train`, keyed by that same `conv_speaker`.
conversations_train's `speaker_id` and surveys_train's `person_id` share the
same p0NN namespace, and all 974 speaker-pairs have a survey row.

CAVEAT: Big-5 is a stable trait, so a person carries the *same* labels into all
of their conversations (verified: zero people vary). Random k-fold CV therefore
puts the same person in both train and test, and the reported accuracy is
optimistic -- it partly measures speaker re-identification. Group the folds by
person (DLATK's --fold_column) if you need a clean generalisation estimate.

Steps (all run when no step flag is passed):
  1. --prepare     add `conv_speaker`, build the `big5_train` outcome table.
  2. --embeddings  roberta-large embeddings, mean-pooled over each speaker's
                   turns within one conversation, into
                   `feat$roberta_la_meL23con$...$conv_speaker`.
  3. --train       10-fold cross-validated ridge, reporting R/r/rho/MSE/MAE per
                   outcome to output/, plus a final model over all 974 units
                   saved to models/.
  4. --predict     apply that model, writing predicted Big-5 per (conversation,
                   speaker) to `feat$p_ridg_big5$conversations_train$conv_speaker`.
"""

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text

### ---- Configuration ----

DATABASE = "memllm"
DB_URL = f"mysql://ssubrahmanya@localhost/{DATABASE}?charset=utf8mb4"

# conversations_train's text column is `text`, not DLATK's default `message`
# (dlaConstants.DEF_MESSAGE_FIELD), so both fields must be passed explicitly.
MESSAGE_TABLE = "conversations_train"
MESSAGE_FIELD = "text"
MESSAGEID_FIELD = "message_id"

# One prediction per (conversation, speaker). DLATK's group (correl) field has to
# be a single column of the message table, so --prepare materialises this one.
GROUP_FIELD = "conv_speaker"
CONVERSATION_FIELD = "conversation_id"
SPEAKER_FIELD = "speaker_id"

SURVEY_TABLE = "surveys_train"
SURVEY_PERSON_FIELD = "person_id"
OUTCOME_TABLE = "big5_train"
OUTCOMES = [
    "personality_openness",
    "personality_conscientiousness",
    "personality_extraversion",
    "personality_agreeableness",
    "personality_stability",  # emotional stability, i.e. reverse-scored neuroticism
]

EMBEDDING_MODEL = "roberta-large"
EMBEDDING_MSG_AGGREGATION = ["mean"]  # over turns, within a (conversation, speaker)
EMBEDDING_WORD_AGGREGATION = ["mean"]  # over word pieces, within a message
EMBEDDING_LAYER_AGGREGATION = ["concatenate"]
EMBEDDING_LAYERS = [23]  # second-to-last of roberta-large's 24 layers
EMBEDDING_NO_CONTEXT = False

ALPHA = 1000
MODEL = f"ridge{ALPHA}"
NFOLDS = 10

# Keep every unit: a gft > 0 would also require a separate 1gram word table.
GROUP_FREQ_THRESH = 0

# Name passed to --predict_regression_to_feats; DLATK prefixes it with
# "p_" + MODEL[:4] (ridge1000 -> "ridg").
PREDICTION_NAME = "big5"

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
DLATK = TUTORIAL_DIR / "dlatk" / "dlatkInterface.py"

# Run dlatkInterface.py under a shim rather than directly. This keeps the vendored
# DLATK checkout unmodified while bridging two version gaps it predates:
#
#   numpy: DLATK imports LinAlgError from `numpy.linalg.linalg`, a private path
#     removed in numpy 2 (regressionPredictor.py:62). Under this venv's numpy 2.5
#     that is an *import-time* failure, so every regression step needs the alias.
#   transformers: addEmbTable imports a batch of legacy model classes behind one
#     `except ImportError: sys.exit(1)` (featureExtractor.py:1252-1263).
#     transformers 5 dropped TransfoXL, so that whole block fails and the
#     embedding step dies claiming torch is missing. Nothing here uses TransfoXL
#     -- it is commented out of DLATK's own MODEL_DICT -- so a placeholder is
#     enough to let the import through.
#
# The sys.path insert is needed because `python -c` puts the cwd on sys.path
# rather than the script's own directory, so `import dlatk` would otherwise miss.
COMPAT_SHIM = """
import pathlib, runpy, sys
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))

import numpy.linalg
sys.modules.setdefault("numpy.linalg.linalg", numpy.linalg)

try:
    import transformers
    for _removed in ("TransfoXLModel", "TransfoXLTokenizer"):
        if not hasattr(transformers, _removed):
            setattr(transformers, _removed, None)
except ImportError:
    pass

runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _emb_feature_table():
    """Reproduce the table name addEmbTable() derives (featureExtractor.py:1327).

    modelNameShort = <first piece>_<2 chars each remaining piece>
                     _[noc_]<2 chars each msg agg>L<layers, L-joined>
                     <2 chars each layer agg>n
    e.g. roberta-large + mean + [23] + concatenate -> roberta_la_meL23con

    Note DLATK builds this from the *message* aggregation, not the word
    aggregation -- they merely happen to both default to "mean". There is no
    "$16to16" extension because --valuefunc defaults to None, so
    createFeatureTable's `if valueFunc:` branch never fires (featureRefiner.py:841).
    """
    pieces = EMBEDDING_MODEL.rsplit("/", maxsplit=1)[-1].split("-")
    short = "_".join([pieces[0]] + [p[:2] for p in pieces[1:]])
    noc = "noc_" if EMBEDDING_NO_CONTEXT else ""
    short += (
        "_" + noc
        + "".join(a[:2] for a in EMBEDDING_MSG_AGGREGATION)
        + "L" + "L".join(str(layer) for layer in EMBEDDING_LAYERS)
        + "".join(a[:2] for a in EMBEDDING_LAYER_AGGREGATION)
        + "n"
    )
    return f"feat${short}${MESSAGE_TABLE}${GROUP_FIELD}"


FEATURE_TABLE = _emb_feature_table()


@dataclass(frozen=True)
class Variant:
    """One feature space and its regressor, over the shared units and outcomes.

    Everything downstream of feature extraction -- the cross-validated fit, the
    saved pickle, the applied predictions -- is identical across feature spaces
    and differs only in these five fields. train() and predict() take a Variant so
    embeddings and n-grams share one code path; only the extractor differs.
    """

    name: str  # stem for the pickle and the output CSVs
    feature_table: str
    model: str  # a key of RegressionPredictor.cvParams, e.g. ridge1000, ridgecv
    prediction_name: str  # DLATK prefixes this with "p_" + model[:4]
    feature_selection: str = ""  # e.g. magic_sauce; "" leaves features untouched

    @property
    def pickle_file(self):
        return REPO_ROOT / "models" / f"{self.name}.pickle"

    @property
    def output_name(self):
        """--output_name is a *prefix*; DLATK appends its own suffixes."""
        return REPO_ROOT / "output" / self.name

    @property
    def accuracy_csv(self):
        """One row per outcome: R, r, rho, R2, MSE, MAE, N, num_features."""
        return self.output_name.parent / f"{self.name}.accuracy_data.csv"

    @property
    def predicted_csv(self):
        """Predicted and true Big-5 per (conversation, speaker)."""
        return self.output_name.parent / f"{self.name}.predicted_data.csv"

    @property
    def prediction_table(self):
        return f"feat$p_{self.model[:4]}_{self.prediction_name}${MESSAGE_TABLE}${GROUP_FIELD}"


EMBEDDING = Variant(
    name=f"big5_{GROUP_FIELD}_{EMBEDDING_MODEL}_L{'L'.join(str(l) for l in EMBEDDING_LAYERS)}_{MODEL}",
    feature_table=FEATURE_TABLE,
    model=MODEL,
    prediction_name=PREDICTION_NAME,
)

### ---- Execution ----


def run(args):
    """Invoke dlatkInterface.py with `args`, under the compat shim."""
    # dlatk/ lives inside this tutorial folder, so run from here.
    command = [PYTHON, "-c", COMPAT_SHIM, DLATK, *args]
    subprocess.run([str(c) for c in command], cwd=TUTORIAL_DIR, check=True)


def prepare():
    """Add the `conv_speaker` group column, then build its Big-5 outcome table."""
    engine = create_engine(DB_URL)
    with engine.begin() as conn:

        # DLATK dedupes messages per group by messageid_field (featureExtractor
        # keeps a `mids` set), so a non-unique id silently drops messages rather
        # than erroring. Fail loudly here instead.
        rows, distinct_ids = conn.execute(
            text(f"SELECT COUNT(*), COUNT(DISTINCT {MESSAGEID_FIELD}) FROM {MESSAGE_TABLE}")
        ).one()
        if rows != distinct_ids:
            raise SystemExit(
                f"{MESSAGE_TABLE}.{MESSAGEID_FIELD} is not unique "
                f"({distinct_ids} distinct values over {rows} rows); DLATK would "
                f"drop {rows - distinct_ids} messages during feature extraction."
            )
        print(f"[{MESSAGE_TABLE}.{MESSAGEID_FIELD}: {rows} unique message ids]")

        # The group field must be one column of the message table. A STORED
        # generated column keeps it in lockstep with its two source columns, and
        # is indexable -- featureExtractor loops `getMessagesForCorrelField` once
        # per group, so the index matters. Adding it is idempotent, and it also
        # survives in information_schema, which createFeatureTable reads to pick
        # the feature table's group_id type (featureRefiner.py:847).
        (has_column,) = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = :db "
                "AND table_name = :tbl AND column_name = :col"
            ),
            {"db": DATABASE, "tbl": MESSAGE_TABLE, "col": GROUP_FIELD},
        ).one()
        if not has_column:
            print(f"[adding generated column {MESSAGE_TABLE}.{GROUP_FIELD}]")
            conn.exec_driver_sql(
                f"ALTER TABLE {MESSAGE_TABLE} "
                f"  ADD COLUMN {GROUP_FIELD} VARCHAR(16) "
                f"    GENERATED ALWAYS AS (CONCAT({CONVERSATION_FIELD}, '_', {SPEAKER_FIELD})) STORED, "
                f"  ADD KEY idx_{MESSAGE_TABLE}_{GROUP_FIELD} ({GROUP_FIELD})"
            )

        # surveys_train is already keyed at (conversation, person) -- the exact
        # unit we model -- so there is nothing to collapse. Confirm that, because
        # a duplicate pair would silently collide on the primary key below.
        pairs, distinct_pairs = conn.execute(
            text(
                f"SELECT COUNT(*), COUNT(DISTINCT {CONVERSATION_FIELD}, {SURVEY_PERSON_FIELD}) "
                f"FROM {SURVEY_TABLE}"
            )
        ).one()
        if pairs != distinct_pairs:
            raise SystemExit(
                f"{SURVEY_TABLE} has {pairs - distinct_pairs} duplicate "
                f"({CONVERSATION_FIELD}, {SURVEY_PERSON_FIELD}) rows; each one is "
                f"supposed to be a single prediction unit."
            )

        print(f"[rebuilding outcome table {OUTCOME_TABLE}]")
        columns = ", ".join(f"{o} DOUBLE NOT NULL" for o in OUTCOMES)
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {OUTCOME_TABLE}")
        conn.exec_driver_sql(
            f"CREATE TABLE {OUTCOME_TABLE} ("
            f"  {GROUP_FIELD} VARCHAR(16) NOT NULL PRIMARY KEY, {columns}"
            f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        # Restrict to pairs that actually speak. surveys_train covers 500
        # conversations but conversations_train only holds 487, and
        # RegressionPredictor.train() takes its group list straight from the
        # outcome table (regressionPredictor.py:577, restrictToGroups defaults to
        # None). getGroupNormsWithZerosFeatsFirst then zero-fills every group
        # missing from the feature table (featureGetter.py:336). So a survey row
        # with no language would train as an all-zero embedding carrying a real
        # Big-5 label, biasing both the fit and the cross-validated accuracy.
        inserted = conn.execute(
            text(
                f"INSERT INTO {OUTCOME_TABLE} ({GROUP_FIELD}, {', '.join(OUTCOMES)}) "
                f"SELECT CONCAT(s.{CONVERSATION_FIELD}, '_', s.{SURVEY_PERSON_FIELD}), "
                f"       {', '.join('s.' + o for o in OUTCOMES)} "
                f"FROM {SURVEY_TABLE} s WHERE EXISTS ("
                f"  SELECT 1 FROM {MESSAGE_TABLE} m "
                f"  WHERE m.{CONVERSATION_FIELD} = s.{CONVERSATION_FIELD} "
                f"    AND m.{SPEAKER_FIELD} = s.{SURVEY_PERSON_FIELD})"
            )
        )

        # Every group DLATK will find in the message table must have an outcome
        # row, or it silently trains on fewer units than the corpus contains.
        (groups,) = conn.execute(
            text(f"SELECT COUNT(DISTINCT {GROUP_FIELD}) FROM {MESSAGE_TABLE}")
        ).one()
        if inserted.rowcount != groups:
            raise SystemExit(
                f"{groups} (conversation, speaker) pairs speak in {MESSAGE_TABLE} but "
                f"only {inserted.rowcount} have a survey row."
            )
        print(
            f"  {inserted.rowcount} (conversation, speaker) units "
            f"({pairs - inserted.rowcount} surveyed pairs never speak, excluded)"
        )

        # Same person, same labels, many conversations -- so a random fold split
        # leaks. Surface the leverage rather than bury it in the docstring.
        (people,) = conn.execute(
            text(f"SELECT COUNT(DISTINCT {SPEAKER_FIELD}) FROM {MESSAGE_TABLE}")
        ).one()
        print(
            f"  spread over {people} distinct people (~{inserted.rowcount / people:.1f} "
            f"units each, identical labels); random {NFOLDS}-fold CV will leak across folds"
        )
    engine.dispose()


def embeddings():
    """roberta-large embeddings, mean-pooled per (conversation, speaker)."""
    run(
        [
            "-d", DATABASE,
            "-t", MESSAGE_TABLE,
            "-c", GROUP_FIELD,
            "--message_field", MESSAGE_FIELD,
            "--messageid_field", MESSAGEID_FIELD,
            "--add_emb_feat",
            "--embedding_model", EMBEDDING_MODEL,
            "--embedding_msg_aggregation", *EMBEDDING_MSG_AGGREGATION,
            "--embedding_word_aggregation", *EMBEDDING_WORD_AGGREGATION,
            "--embedding_layer_aggregation", *EMBEDDING_LAYER_AGGREGATION,
            "--embedding_layers", *[str(layer) for layer in EMBEDDING_LAYERS],
        ]
    )


def _regression_args(variant):
    """The data, features and model every regression invocation names alike."""
    return [
        "-d", DATABASE,
        "-t", MESSAGE_TABLE,
        "-c", GROUP_FIELD,
        "--message_field", MESSAGE_FIELD,
        "--messageid_field", MESSAGEID_FIELD,
        "-f", variant.feature_table,
        "--outcome_table", OUTCOME_TABLE,
        "--outcomes", *OUTCOMES,
        "--group_freq_thresh", str(GROUP_FREQ_THRESH),
        "--model", variant.model,
        *(["--feature_selection", variant.feature_selection] if variant.feature_selection else []),
    ]


def train(variant):
    """10-fold cross-validated fit, plus a final all-unit model to the pickle.

    Two dlatkInterface runs, not one. They could be a single invocation -- DLATK
    happily does --nfold_regression and --train_regression together -- but
    --output_name is what makes the metrics persistent, and passing it alongside
    --train_regression also sets `saveFeatures = True` (dlatkInterface.py:1890).
    That np.savetxt's the whole dense design matrix once per outcome
    (regressionPredictor.py:658): five near-identical CSVs, to no purpose. So the
    evaluation run takes --output_name and the fit that saves the pickle does not.
    The price is loading the feature table twice.

    Run 1 (--nfold_regression) writes variant.accuracy_csv and .predicted_csv. It
    reports cross-validated accuracy and saves no model: testControlCombos never
    assigns self.regressionModels, and prints "!!SAVING MODELS NOT IMPLEMENTED!!"
    if asked.

    Run 2 (--train_regression) fits one model per outcome over all 974 units and
    stores all five in a single pickle, keyed by outcome name. This is the model
    predict() later applies -- not any of run 1's fold models.
    """
    variant.output_name.parent.mkdir(parents=True, exist_ok=True)
    variant.pickle_file.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            *_regression_args(variant),
            "--nfold_regression",
            "--folds", str(NFOLDS),
            # --csv redirects the metrics dict from a stdout pprint into the
            # accuracy CSV; --pred_csv turns on savePredictions and writes the
            # prediction CSV. Both suffix --output_name (dlatkInterface.py:1916-1929).
            "--csv",
            "--pred_csv",
            "--output_name", variant.output_name,
        ]
    )
    print(f"[metrics -> {variant.accuracy_csv}]\n[fold predictions -> {variant.predicted_csv}]")

    run([*_regression_args(variant), "--train_regression", "--save", "--picklefile", variant.pickle_file])


def predict(variant):
    """Apply the saved model, writing predicted Big-5 per unit to a feature table."""
    run(
        [
            *_regression_args(variant),
            "--predict_regression_to_feats", variant.prediction_name,
            "--load",
            "--picklefile", variant.pickle_file,
        ]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help=f"Step 1: add {MESSAGE_TABLE}.{GROUP_FIELD}, build the {OUTCOME_TABLE} outcome table",
    )
    parser.add_argument(
        "-e", "--embeddings",
        action="store_true",
        help=f"Step 2: extract {EMBEDDING_MODEL} embeddings into {FEATURE_TABLE}",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=f"Step 3: {NFOLDS}-fold cross-validated {MODEL}, writing "
             f"{EMBEDDING.accuracy_csv.name} and saving {EMBEDDING.pickle_file.name}",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help=f"Step 4: apply the pickle, writing {EMBEDDING.prediction_table}",
    )
    args = parser.parse_args()

    # No flags means run the whole pipeline.
    if not (args.prepare or args.embeddings or args.train or args.predict):
        args.prepare = args.embeddings = args.train = args.predict = True

    if args.prepare:
        prepare()
    if args.embeddings:
        embeddings()
    if args.train:
        train(EMBEDDING)
    if args.predict:
        predict(EMBEDDING)
        print(f"\n[predictions written to {DATABASE}.{EMBEDDING.prediction_table}]")

