from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

PERFORMANCE_COLUMNS = [
    "outcome",
    "mae",
    "mse",
    "N",
    "num_features",
    "R2",
    "r",
    "r_p",
    "r_folds",
    "r_p_folds",
    "se_r_folds",
    "train_size",
    "test_size",
]

TRAIT_TO_OUTCOME = {
    "agreeableness": "agreeable_score",
    "conscientiousness": "conscientious_score",
    "extraversion": "extravert_score",
    "neuroticism": "neurotic_score",
    "openness": "openness_score",
}


def load_accuracy_data(dataset, output_name):

    path = Path(f"{dataset}/results")
    performance = pd.read_csv(path / f"{output_name}.accuracy_data.csv", skiprows=1)
    if "mae" not in performance.columns:
        # this file's DLATK header block additionally has an abbreviated "w/ lang."
        # mini-header ahead of the real one
        performance = pd.read_csv(path / f"{output_name}.accuracy_data.csv", skiprows=3)
    # DLATK batches outcomes and re-emits a header per batch, which can survive as a
    # bogus data row -- drop any recurrence.
    performance = performance[performance["outcome"] != "outcome"].reset_index(drop=True)
    performance = performance[PERFORMANCE_COLUMNS]
    numeric_columns = [c for c in PERFORMANCE_COLUMNS if c != "outcome"]
    performance[numeric_columns] = performance[numeric_columns].astype(float)
    performance["dataset"] = dataset

    return performance


def get_correlations(
    engine,
    feature_table,
    meta_table="feat$meta_1gram$msg_essays_v9v11$person_id",
    outcome_table="outcomes_v9v11_person",
    trait_to_outcome=TRAIT_TO_OUTCOME,
    group_freq_thresh=100,
):

    query = f'''
        SELECT group_id AS person_id, value AS ntokens
        FROM {meta_table}
        WHERE feat='_total1grams'
    '''
    meta = pd.read_sql(query, engine)

    query = f"SELECT * FROM {outcome_table}"
    outcome_table = pd.read_sql(query, engine)

    query = f"SELECT group_id AS person_id, feat, group_norm FROM {feature_table}"
    features = pd.read_sql(query, engine)
    features = (
        features.pivot_table(
            index="person_id",
            columns="feat",
            values="group_norm"
        )
        .reset_index()
    )

    df = pd.merge(
        outcome_table.merge(
            meta,
            on="person_id"
        ),
        features,
        on="person_id"
    )

    df = df[df["ntokens"] >= group_freq_thresh]

    rows = []
    for trait, outcome in trait_to_outcome.items():
        res = pearsonr(df[trait], df[outcome])
        low, high = res.confidence_interval()
        rows.append(
            {
                "trait": trait,
                "outcome": outcome,
                "r": res.statistic,
                "p": res.pvalue,
                "CI.low": low,
                "CI.high": high,
                "N": len(df)
            }
        )

    return pd.DataFrame(rows)
