"""Predict the Big-5 inventory from `memllm.conversations_train` n-grams.

The n-gram counterpart to commands.py: same units, same outcomes, same folds,
same regression code path -- only the feature space changes, from mean-pooled
roberta-large embeddings to 1-to-3 grams. Follows
https://dlatk.github.io/dlatk/tutorials/tut_pred.html.

Everything downstream of feature extraction is shared, so this module contributes
exactly two things: the NGRAM Variant, and the ngrams() extractor that builds its
feature table. prepare(), train() and predict() are commands.py's, unmodified.

One training unit per (conversation, speaker), keyed by the `conv_speaker` group
column. That column, and the `big5_train` outcome table keyed by it, are built by
commands.prepare() -- see the docstring there for why DLATK cannot read
`surveys_train` directly.

Feature space. Raw 1/2/3-grams over 974 units give 235,981 distinct features, far
more than ridge can fit without the regularisation path collapsing onto noise
terms seen in one conversation. --feat_occ_filter keeps only features occurring
in more than P_OCC of the groups, dropping that to 901 at P_OCC = 0.05 -- the
same order as the 704 features of the pretrained OCEAN pickle in apply_pickle.py.
DLATK writes the survivors to a new table suffixed with the threshold (`$0_05`),
leaving the unfiltered table in place.

CAVEAT: inherited from commands.py -- Big-5 is a stable trait, so a person carries
identical labels into all of their conversations. Random k-fold CV puts the same
person in both train and test, so the reported accuracy partly measures speaker
re-identification. n-grams make that *worse* than embeddings do: an idiolect
(a rare trigram a person reuses) is a near-unique speaker fingerprint. Group the
folds by person (DLATK's --fold_column) for a clean generalisation estimate.

Steps (all run when no step flag is passed):
  1. --prepare       add `conv_speaker`, build the `big5_train` outcome table.
                     Shared with commands.py; a no-op once it has run.
  2. --ngrams        extract 1/2/3-grams, combine into one `1to3gram` table, then
                     occurrence-filter it to `...$0_05`.
  3. --train_ngram   10-fold cross-validated ridgecv, reporting R/r/rho/MSE/MAE
                     per outcome to output/, plus a final model over all 974
                     units saved to models/.
  4. --predict_ngram apply that model, writing predicted Big-5 per (conversation,
                     speaker) to `feat$p_ridg_big5ng$conversations_train$conv_speaker`.
"""

import argparse

# On merge into commands.py, drop this import: every name is already in that module.
from commands import (
    DATABASE,
    GROUP_FIELD,
    GROUP_FREQ_THRESH,
    MESSAGE_TABLE,
    MESSAGE_FIELD,
    MESSAGEID_FIELD,
    NFOLDS,
    Variant,
    predict,
    prepare,
    run,
    train,
)

### ---- Configuration ----

NGRAMS = ["1", "2", "3"]
FEATURE_NAME = "1to3gram"

# Keep features occurring in more than 5% of the 974 units: 235,981 -> 901.
# DLATK labels the filtered table with the threshold, '.' -> '_'
# (featureRefiner.createTableWithRemovedFeats:286).
P_OCC = 0.05
P_OCC_LABEL = str(P_OCC).replace(".", "_")

BASE_FEATURE_TABLE = f"feat${FEATURE_NAME}${MESSAGE_TABLE}${GROUP_FIELD}"
FEATURE_TABLE = f"{BASE_FEATURE_TABLE}${P_OCC_LABEL}"

# RidgeCV picks alpha per outcome by internal CV. commands.py can hard-code
# ridge1000 because roberta's 1024 dense dimensions are on a known scale; the
# right penalty for a sparse, occurrence-filtered n-gram matrix is not knowable
# up front. Set feature_selection="magic_sauce" for DLATK's standard n-gram
# pipeline (occurrence threshold -> SelectFwe -> randomized PCA); it is left off
# here so the learned coefficients stay one-per-n-gram and remain readable.
#
# prediction_name is "big5ng", not commands.py's "big5", so the embedding and
# n-gram predictions sit side by side in the database rather than overwrite.
NGRAM = Variant(
    name=f"big5_{GROUP_FIELD}_{FEATURE_NAME}_p{P_OCC_LABEL}_ridgecv",
    feature_table=FEATURE_TABLE,
    model="ridgecv",
    prediction_name="big5ng",
    feature_selection="",
)

### ---- Execution ----


def ngrams():
    """1/2/3-grams per (conversation, speaker), combined, then occurrence-filtered.

    dlatkInterface applies these in order within one invocation: extract each n
    (:1038), combine the three tables into `1to3gram` (:1319), filter that
    (:1343). Each stage feeds the next through args.feattable, so the whole chain
    is one command rather than three.

    GROUP_FREQ_THRESH stays 0, matching commands.py so the two models are scored
    over the same 974 rows. A non-zero threshold is now *possible* -- the 1gram
    word table it needs, feat$1gram$conversations_train$conv_speaker, is a
    by-product of this step -- but the corpus is short (52 words for the quietest
    speaker, 233 on average), so DLATK's default of 1000 would empty the set.
    """
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
            "--feat_occ_filter",
            "--set_p_occ", str(P_OCC),
            "--group_freq_thresh", str(GROUP_FREQ_THRESH),
        ]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help=f"Step 1: add {MESSAGE_TABLE}.{GROUP_FIELD}, build the Big-5 outcome table",
    )
    parser.add_argument(
        "--ngrams",
        action="store_true",
        help=f"Step 2: extract {'/'.join(NGRAMS)}-grams into {FEATURE_TABLE}",
    )
    parser.add_argument(
        "--train_ngram",
        action="store_true",
        help=f"Step 3: {NFOLDS}-fold cross-validated {NGRAM.model}, writing "
             f"{NGRAM.accuracy_csv.name} and saving {NGRAM.pickle_file.name}",
    )
    parser.add_argument(
        "--predict_ngram",
        action="store_true",
        help=f"Step 4: apply the pickle, writing {NGRAM.prediction_table}",
    )
    args = parser.parse_args()

    # No flags means run the whole pipeline.
    if not (args.prepare or args.ngrams or args.train_ngram or args.predict_ngram):
        args.prepare = args.ngrams = args.train_ngram = args.predict_ngram = True

    if args.prepare:
        prepare()
    if args.ngrams:
        ngrams()
    if args.train_ngram:
        train(NGRAM)
    if args.predict_ngram:
        predict(NGRAM)
        print(f"\n[predictions written to {DATABASE}.{NGRAM.prediction_table}]")
