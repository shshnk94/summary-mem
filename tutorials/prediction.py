from __future__ import annotations

import os
import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import InContextMemory, SummaryMemory
from summary_mem.prompts.summarization import QUESTIONNAIRE_SUMMARY, SUMMARY

# --summary_prompt -> (prompt template, table it's stored under).
SUMMARY_PROMPTS = {
    "questionnaire": (QUESTIONNAIRE_SUMMARY, "summaries"),
    "plain": (SUMMARY, "summaries_plain"),
}

# DLATK feature-table name prefixes, keyed by base_tag (--memory, plus
# --summary_prompt when --memory=summary). {corpus} and the turn tag are filled
# in at the call site (see corpus_tag(), turn_tag()).
BIG5_FEATURE_TABLES = {
    "in_context": "feat$bfi${corpus}$person_id$ic$gpt4omini",
    "summary_plain": "feat$bfi${corpus}$person_id$sumplain$gpt4omini",
    "summary_questionnaire": "feat$bfi${corpus}$sumquest$gpt4omini",
}

def corpus_tag(namespace: str) -> str:
    """MySQL-identifier-safe tag for --namespace, keeping table names under the 64-char limit."""
    return re.sub(r"[^a-zA-Z0-9]", "", namespace)[:10]


def turn_tag(num_recent_turns: int | None, turn_stride: int) -> str:
    """Short, MySQL-identifier-safe tag for which turns were kept."""
    return f"rt{num_recent_turns}" if num_recent_turns is not None else f"s{turn_stride}"

load_dotenv()

# Trait order for the LLM's JSON response; --outcomes lines up with this order.
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# {context} is either a SummaryMemory summary or the user's raw turns -- both are
# a plain text blob, so one prompt covers both modes.
PERSONALITY_ASSESSMENT_PROMPT = (
    "Your goal is to assess a user's Big Five personality traits, given a summary of "
    "their language, or their language directly.\n\n"

    "The Big Five traits and the facets that define them are:\n"

    "* Openness:\n"
    "- Imagination: vivid imagination, daydreaming\n"
    "- Artistic Interests: appreciation for art, beauty, and poetry\n"
    "- Emotionality: aware of and in touch with one's own feelings\n"
    "- Adventurousness: preference for novelty and variety over routine\n"
    "- Intellect: curiosity, enjoyment of abstract or theoretical discussion\n"
    "- Liberalism: willingness to reconsider one's own beliefs\n"

    "* Conscientiousness:\n"
    "- Self-Efficacy: confidence in one's own ability to get things done\n"
    "- Orderliness: preference for structure and organization\n"
    "- Dutifulness: strict adherence to obligations and ethical principles\n"
    "- Achievement-Striving: drive, ambition, work ethic\n"
    "- Self-Discipline: follow-through despite distraction or difficulty\n"
    "- Cautiousness: thinking carefully before acting rather than acting on impulse\n"

    "* Extraversion:\n"
    "- Friendliness: warmth, interest in close relationships\n"
    "- Gregariousness: preference for the company of others, enjoys crowds\n"
    "- Assertiveness: taking the lead, being direct or forceful\n"
    "- Activity Level: a fast pace and high energy in daily life\n"
    "- Excitement-Seeking: craving stimulation and risk, seeking thrills\n"
    "- Cheerfulness: joy, enthusiasm, optimism\n"

    "* Agreeableness:\n"
    "- Trust: assuming others are honest and well-intentioned\n"
    "- Morality: frankness and sincerity, dislike of manipulation or deception\n"
    "- Altruism: concern for others' welfare, willingness to help\n"
    "- Cooperation: avoids conflict, defers rather than competes\n"
    "- Modesty: humble, downplays one's own achievements\n"
    "- Sympathy: tender-hearted, moved by others' misfortune\n"

    "* Neuroticism (score high when these facets show up a lot):\n"
    "- Anxiety: worry, tension, fearfulness\n"
    "- Anger: frustration or irritation in response to setbacks\n"
    "- Depression: sadness, hopelessness, discouragement\n"
    "- Self-Consciousness: sensitivity to social judgment, shyness, embarrassment\n"
    "- Immoderation: difficulty resisting cravings and urges\n"
    "- Vulnerability: how well they cope with stress or difficulty\n\n"

    "Instructions:\n"
    "Score each trait on a continuous 1-5 scale (1 = very low, 3 = moderate, 5 = very high), based only on evidence "
    "present in the input -- direct statements, described behavior, or patterns in language and tone. "
    "Never fabricate or infer evidence the input does not support.\n\n"

    "Respond with a single JSON object:\n"
    "{{\"openness\": <number>, \"conscientiousness\": <number>, \"extraversion\": <number>, "
    "\"agreeableness\": <number>, \"neuroticism\": <number>}}\n"
    "Always include all five keys, even when evidence for a trait is limited -- give your best "
    "estimate rather than omitting the key. Output only the JSON -- no markdown formatting, no "
    "surrounding text.\n\n"

    "Here is what is known about the user {user}, from a summary of their language or "
    "their language directly:\n\n"
    "{context}\n\n"
    "Assess {user}'s Big Five personality trait scores (1-5) as JSON."
)


def model_slug(model: str) -> str:
    """Filesystem-safe tag for a model name, e.g. 'openai/gpt-4o-mini' -> 'gpt4omini'."""
    return re.sub(r"[^a-zA-Z0-9]", "", model.rsplit("/", 1)[-1])


def read_conversation_turns(
    engine,
    conversation_id: str,
    message_table: str,
    conversation_field: str,
    user_field: str,
    time_id: str,
) -> pd.DataFrame:

    query = (
        f"SELECT {conversation_field} AS conversation_id, {user_field} AS user_id, "
        f"{time_id} AS turn_id, message FROM {message_table} "
        f"WHERE {conversation_field} = '{conversation_id}' "
        f"ORDER BY {time_id}"
    )

    turns = pd.read_sql(query, engine)
    return turns


def restrict_to_users(
    engine,
    message_table: str,
    conversation_field: str,
    user_field: str,
    user_ids: list[str],
) -> np.ndarray:
    """Expand a fixed --user_field sample into the --conversation_field ids the
    rest of the pipeline stores/assesses over (e.g. ds4ud's wave_id splits one
    user into several conversations)."""
    ids = "', '".join(user_ids)
    query = (
        f"SELECT DISTINCT {conversation_field} FROM {message_table} "
        f"WHERE {user_field} IN ('{ids}') AND {conversation_field} IS NOT NULL"
    )
    return pd.read_sql(query, engine)[conversation_field].to_numpy()


def read_outcomes(
    engine,
    user_ids: np.ndarray,
    outcome_table: str,
    user_field: str,
    outcomes: list[str],
) -> pd.DataFrame:

    user_ids = "', '".join(user_ids.tolist())
    query = (
        f"SELECT {user_field}, {', '.join(outcomes)} FROM {outcome_table} "
        f"WHERE {user_field} IN ('{user_ids}') "
        f"ORDER BY {user_field}"
    )
    outcomes_df = pd.read_sql(query, engine)
    outcomes_df[user_field] = outcomes_df[user_field].astype(str)

    return outcomes_df


async def store_conversations(
    memory: SummaryMemory | InContextMemory,
    conversation_ids: np.ndarray,
    message_table: str,
    conversation_field: str,
    user_field: str,
    time_id: str,
    engine: Engine,
    max_concurrency: int = 8,
    num_recent_turns: int | None = None,
    turn_stride: int = 2,
) -> None:

    semaphore = asyncio.Semaphore(max_concurrency)
    progress = tqdm(total=len(conversation_ids), desc="storing conversations")

    async def store_conversation(conversation_id) -> None:

        async with semaphore:
            turns = await asyncio.to_thread(
                read_conversation_turns,
                engine,
                conversation_id,
                message_table,
                conversation_field,
                user_field,
                time_id
            )

            turns = turns.sort_values("turn_id")
            if num_recent_turns is not None:
                # keep each speaker's most recent num_recent_turns, in order
                turns = turns.groupby("user_id").tail(num_recent_turns)
            else:
                # keep every turn_stride-th turn per speaker, in order
                turn_index_within_speaker = turns.groupby("user_id").cumcount()
                turns = turns[turn_index_within_speaker % turn_stride == 0]

            for index, turn in turns.iterrows():
                await asyncio.to_thread(
                    memory.update,
                    conversation_id=str(turn["conversation_id"]),
                    speaker_id=str(turn["user_id"]),
                    turn_text=turn["message"],
                )
        progress.update(1)

    try:
        await asyncio.gather(*(store_conversation(cid) for cid in conversation_ids))
    finally:
        progress.close()


def build_user_context(user_id: str, memory: SummaryMemory | InContextMemory) -> str:
    """Aggregate a user's context across every conversation they appear in."""
    contexts = memory.recall_all(user_id)
    context = "\n\n".join(f"Excerpt {i + 1}:\n{c}" for i, c in enumerate(contexts))
    return context


def assess_personality(
    chat_client: OpenAI,
    model: str,
    user_id: str,
    context: str,
) -> dict[str, float]:
    response = chat_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PERSONALITY_ASSESSMENT_PROMPT.format(
                    user=user_id,
                    context=context,
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    return {trait: float(scores[trait]) for trait in TRAITS}


def to_feature_table(assessments: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide per-user trait columns into DLATK's long feature-table format
    (https://dlatk.github.io/dlatk/tutorials/tut_feat_tables.html): one row per
    (group_id, feat). group_norm is set equal to value rather than DLATK's usual
    value-divided-by-group-sum, since the five traits aren't parts of a whole;
    it's kept only because downstream DLATK tooling expects the column to exist.
    """
    long = assessments.melt(
        id_vars="user_id",
        value_vars=[f"assessed_{trait}" for trait in TRAITS],
        var_name="feat",
        value_name="value",
    )
    long["feat"] = long["feat"].str.removeprefix("assessed_")
    long = long.rename(columns={"user_id": "group_id"})
    long["group_norm"] = long["value"]
    long = long[["group_id", "feat", "value", "group_norm"]]

    return long


def store_assessments(
    assessments: pd.DataFrame,
    results_dir: Path,
    run_tag: str,
    feature_table_name: str,
    model: str,
    engine: Engine,
) -> None:
    """Persist a run's assessments to results_dir's CSV and its BIG5 feature table."""
    assessments.to_csv(
        results_dir / f"assessments_{run_tag}_{model_slug(model)}.csv",
        index=False,
    )

    feature_table = to_feature_table(assessments)
    feature_table.to_sql(feature_table_name, engine, if_exists="replace", index=False)


def evaluate(
    assessments: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    user_field: str,
    outcomes: list[str],
) -> None:
    """Score assessments against self-reported outcomes (Pearson r per trait).

    Outcomes are averaged per user_field first, since some corpora (e.g. CANDOR)
    survey a user more than once; a no-op when there's already one row per user.
    """
    counts = outcomes_df.groupby(user_field).size()
    if (counts > 1).any():
        outcomes_df = outcomes_df.groupby(user_field, as_index=False)[outcomes].mean()

    comparison = pd.merge(
        assessments,
        outcomes_df,
        left_on="user_id",
        right_on=user_field,
        how="left",
    )

    for trait, outcome in zip(TRAITS, outcomes):
        valid = comparison[[f"assessed_{trait}", outcome]].dropna()
        if len(valid) < 2:
            print(f"{trait:18s} ({outcome}) Pearson r = n/a  (n={len(valid)}; need >=2 points)")
            continue
        r, p = pearsonr(valid[f"assessed_{trait}"], valid[outcome])
        print(f"{trait:18s} ({outcome}) Pearson r = {r:.3f}  (n={len(valid)})  p-value = {p:.3e}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # MySQL database and table names for the corpus (defaults: ds4ud).
    parser.add_argument("--database", default="ssubrahmanya", help="MySQL database holding the corpus.")
    parser.add_argument("--message_table", default="msg_essays_v9v11", help="MySQL table holding the messages.")
    parser.add_argument("--outcome_table", default="outcomes_v9v11", help="MySQL table holding the outcomes.")

    # Column names, for corpora that don't share ds4ud's schema (e.g. CANDOR).
    parser.add_argument(
        "--conversation_field",
        default="wave_id",
        help="message table's conversation column (default: %(default)s)."
    )

    parser.add_argument(
        "--user_field",
        default="person_id",
        help="message_table's user/person column (default: %(default)s).",
    )

    parser.add_argument(
        "--turn_field",
        default="startdate",
        help="message_table's column giving each row's order within a conversation (default: %(default)s).",
    )

    parser.add_argument(
        "--group_freq_thresh", 
        type=int, 
        default=100,
        help="Minimum words per users for the user to be considered for sampling.",
    )
    
    parser.add_argument(
        "--outcomes", nargs=5, metavar=tuple(t.upper() for t in TRAITS),
        default=["openness_score", "conscientious_score", "extravert_score", "agreeable_score", "neurotic_score"],
        help="outcome_table's trait columns.",
    )
    # Sampling for testing.
    parser.add_argument(
        "--sample", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Sample num_samples users for analysis"
        )
    
    parser.add_argument(
        "-n", 
        type=int, 
        dest="num_samples", 
        default=100, 
        help="Number of participants to sample."
    )
    
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Write the sampled --user_field ids to results.",
    )

    # OpenRouter/OpenAI API and model.
    parser.add_argument(
        "--url", 
        default="https://openrouter.ai/api/v1", 
        help="OpenRouter API base URL"
    )

    parser.add_argument(
        "--model", 
        default="openai/gpt-4o-mini", 
        help="LLM model to use"
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max number of conversations summarized concurrently."
    )

    # Memory mechanism and model.
    parser.add_argument(
        "--memory",
        choices=["summary", "in_context"],
        default="summary",
        help="'summary' uses summary-mem's rolling summaries; 'in_context' puts every turn in the same prompt.",
    )

    parser.add_argument(
        "--memorydb", 
        default="ssubrahmanya", 
        help="MySQL database to store rolling summaries."
    )
    
    parser.add_argument(
        "--summary_prompt",
        choices=list(SUMMARY_PROMPTS),
        default="questionnaire",
        help="which summarization prompt SummaryMemory uses"
    )

    parser.add_argument(
        "--namespace",
        default="ds4ud",
        help="Scopes summaries in the shared MySQL table to this corpus, so they can't collide "
        "with an unrelated pipeline/corpus writing to the same --memorydb/table (default: %(default)s).",
    )

    parser.add_argument(
        "--num_recent_turns",
        type=int,
        default=None,
        help="If set, keep only each speaker's most recent num_recent_turns turns per conversation.",
    )
    
    parser.add_argument(
        "--turn_stride",
        type=int,
        default=2,
        help="Keep only every turn_stride-th turn per speaker, in order (default: %(default)s).",
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory to store assessment results."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    rng = np.random.default_rng(seed=42)
    engine = create_engine(
        f"mysql://ssubrahmanya@localhost/{args.database}?charset=utf8mb4",
        connect_args={"read_default_file": "~/.my.cnf"},
    )
    args.results_dir.mkdir(exist_ok=True)

    if args.sample:

        # ORDER BY makes eligible's row order (and thus rng.choice()'s draw) a
        # deterministic function of the data instead of MySQL's unspecified
        # SELECT DISTINCT/plain-SELECT row order, so a fixed seed actually
        # reproduces the same sample across runs.
        query = f"SELECT * FROM {args.outcome_table} ORDER BY {args.user_field}"
        outcomes = pd.read_sql(query, engine)
        outcomes = outcomes[[args.user_field] + args.outcomes]

        query = (
            f"SELECT DISTINCT {args.user_field}, {args.conversation_field} FROM {args.message_table} "
            f"ORDER BY {args.user_field}, {args.conversation_field}"
        )
        conversations = pd.read_sql(query, engine)

        word_table = f"feat$meta_1gram${args.message_table}${args.user_field}"
        query = (
            f"SELECT group_id AS {args.user_field}, value FROM `{word_table}` "
            f"WHERE feat = '_total1grams' ORDER BY group_id"
        )
        token_counts = pd.read_sql(query, engine)

        df = (
            outcomes
            .merge(
                conversations,
                on=args.user_field
            )
            .merge(
                token_counts,
                on=args.user_field
            )
        )

        eligible = np.sort(
            df.loc[
                df[args.outcomes].notna().all(axis=1)
                & df[args.conversation_field].notna()
                & (df["value"] >= args.group_freq_thresh),
                args.user_field,
            ].unique()
        )

        users = rng.choice(
            eligible,
            size=min(
                args.num_samples,
                len(eligible)
            ),
            replace=False
        )

    else:

        query = f'''
            SELECT DISTINCT {args.user_field}
            FROM {args.message_table}
            WHERE {args.user_field} IS NOT NULL
            ORDER BY {args.user_field}
        '''
        users = pd.read_sql(query, engine)
        users = users[args.user_field].astype(str).to_numpy()

    if args.sample_only:
        sample_path = args.results_dir / f"sample_{args.user_field}.csv"
        pd.DataFrame({args.user_field: users}).to_csv(sample_path, index=False)
        print(f"Wrote {len(users)} sampled ids to {sample_path}")
        raise SystemExit(0)

    conversation_ids = restrict_to_users(
        engine, 
        args.message_table, 
        args.conversation_field, 
        args.user_field, 
        users
    )

    client = OpenAI(
        base_url=args.url,
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # tag for the feature-table/run names; the shared summaries table is scoped
    # by args.namespace directly (see SummaryMemory(namespace=...) below), so
    # this corpus's rows can never collide with another pipeline/corpus writing
    # to the same --memorydb/table_name
    corpus = corpus_tag(args.namespace)

    if args.memory == "summary":
        prompt_template, table_name = SUMMARY_PROMPTS[args.summary_prompt]
        memory = SummaryMemory(
            client,
            db_name=args.memorydb,
            model=args.model,
            prompt_template=prompt_template,
            table_name=table_name,
            namespace=args.namespace,
        )

        # wipe only this run's conversations -- the MySQL store is shared across runs
        with memory.db.engine.begin() as conn:
            conn.execute(
                memory.db.summaries.delete().where(
                    memory.db.summaries.c.namespace == memory.db.namespace,
                    memory.db.summaries.c.conversation_id.in_(
                        [str(cid) for cid in conversation_ids]
                    ),
                )
            )
    else:
        memory = InContextMemory()

    asyncio.run(
        store_conversations(
            memory,
            conversation_ids,
            args.message_table,
            args.conversation_field,
            args.user_field,
            args.turn_field,
            engine,
            args.max_concurrency,
            args.num_recent_turns,
            args.turn_stride,
        )
    )

    assessments = []
    for user_id in tqdm(users, desc="assessing personality traits"):

        context = build_user_context(user_id, memory)
        assessment = assess_personality(
            client,
            args.model,
            user_id,
            context,
        )

        assessments.append(
            {
                "user_id": user_id,
                **{f"assessed_{trait}": assessment[trait] for trait in TRAITS},
            }
        )

    # fold in prompt variant, corpus, and turn-selection so no run overwrites another's results
    base_tag = f"{args.memory}_{args.summary_prompt}" if args.memory == "summary" else args.memory
    tag = turn_tag(args.num_recent_turns, args.turn_stride)
    run_tag = f"{base_tag}_{corpus}_{tag}"
    feature_table_name = f"{BIG5_FEATURE_TABLES[base_tag].format(corpus=corpus)}${tag}"

    assessments = pd.DataFrame(assessments)
    store_assessments(assessments, args.results_dir, run_tag, feature_table_name, args.model, engine)

    memory.close()

    outcomes_df = read_outcomes(
        engine,
        users,
        args.outcome_table,
        args.user_field,
        args.outcomes,
    )
    evaluate(assessments, outcomes_df, args.user_field, args.outcomes)
