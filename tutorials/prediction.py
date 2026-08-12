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
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import BatchSummaryMemory, ConversationBatchSummaryMemory, InContextMemory, SummaryMemory

from corpus import eligible_users

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


def read_speaker_turns(
    engine,
    user_id: str,
    message_table: str,
    conversation_field: str,
    user_field: str,
    turn_id: str,
) -> pd.DataFrame:

    query = text(
        f"SELECT {conversation_field}, {user_field}, {turn_id}, message FROM {message_table} "
        f"WHERE {user_field} = :user_id "
        f"ORDER BY {turn_id}"
    )

    return pd.read_sql(query, engine, params={"user_id": user_id})



def restrict_to_users(
    engine,
    message_table: str,
    conversation_field: str,
    user_field: str,
    user_ids: list[str],
) -> np.ndarray:
    """Expand a --user_field sample into the --conversation_field ids the rest of
    the pipeline operates over (a user may map to several conversations, e.g.
    ds4ud's wave_id)."""
    query = text(
        f"SELECT DISTINCT {conversation_field} FROM {message_table} "
        f"WHERE {user_field} IN :user_ids AND {conversation_field} IS NOT NULL"
    ).bindparams(bindparam("user_ids", expanding=True))

    return pd.read_sql(query, engine, params={"user_ids": list(user_ids)})[conversation_field].to_numpy()


def read_outcomes(
    engine,
    user_ids: np.ndarray,
    outcome_table: str,
    user_field: str,
    outcomes: list[str],
) -> pd.DataFrame:

    query = text(
        f"SELECT {user_field}, {', '.join(outcomes)} FROM {outcome_table} "
        f"WHERE {user_field} IN :user_ids "
        f"ORDER BY {user_field}"
    ).bindparams(bindparam("user_ids", expanding=True))

    outcomes_df = pd.read_sql(query, engine, params={"user_ids": user_ids.tolist()})
    outcomes_df[user_field] = outcomes_df[user_field].astype(str)

    return outcomes_df


async def build_memory(
    memory: SummaryMemory | InContextMemory | BatchSummaryMemory | ConversationBatchSummaryMemory,
    user_ids: np.ndarray,
    message_table: str,
    conversation_field: str,
    user_field: str,
    turn_id: str,
    engine: Engine,
    max_concurrency: int = 8,
    turn_proportion: float | None = None,
    earliest: bool = False,
    turn_stride: int | None = None,
) -> None:

    semaphore = asyncio.Semaphore(max_concurrency)
    progress = tqdm(total=len(user_ids), desc="storing speakers")

    async def store_speaker(user_id) -> None:

        async with semaphore:
            turns = await asyncio.to_thread(
                read_speaker_turns,
                engine,
                user_id,
                message_table,
                conversation_field,
                user_field,
                turn_id
            )

            turns = turns.sort_values(turn_id)
            if turn_proportion is not None:
                # keep_n across ALL of this speaker's sampled conversations, at least 1
                keep_n = max(1, round(len(turns) * turn_proportion))
                turns = turns.head(keep_n) if earliest else turns.tail(keep_n)
            elif turn_stride is not None:
                # keep every turn_stride-th turn per conversation, in order
                turn_index_within_conversation = turns.groupby(conversation_field).cumcount()
                turns = turns[turn_index_within_conversation % turn_stride == 0]
            # else: neither is set -- consume every turn

            for index, turn in turns.iterrows():
                await asyncio.to_thread(
                    memory.update,
                    conversation_id=str(turn[conversation_field]),
                    speaker_id=str(turn[user_field]),
                    turn_text=turn["message"],
                )
        progress.update(1)

    try:
        await asyncio.gather(*(store_speaker(uid) for uid in user_ids))
    finally:
        progress.close()


def build_user_context(
    user_id: str,
    memory: SummaryMemory | InContextMemory | BatchSummaryMemory | ConversationBatchSummaryMemory,
) -> str:
    """Aggregate a user's context across every conversation they appear in."""
    contexts = memory.recall_speaker(user_id)
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
    """Reshape into DLATK's long feature-table format (one row per group_id/feat).
    group_norm is set equal to value -- the five traits aren't parts of a whole --
    but kept because downstream DLATK tooling expects the column to exist.
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
    base_tag: str,
    corpus: str,
    model_slug: str,
    tag: str,
    engine: Engine,
) -> None:
    """Persist a run's assessments to results_dir's CSV and its BIG5 feature table."""
    assessments.to_csv(
        results_dir / f"assessments_{run_tag}_{model_slug}.csv",
        index=False,
    )

    # short DLATK-style abbreviation for base_tag, keeping the feature-table name
    # under MySQL's 64-char identifier limit
    table_abbrev = {
        "in_context": "ic",
        "summary_plain": "sumplain",
        "batch_summary": "batchsum",
        "conversation_batch_summary": "convbatchsum",
        "mem0": "mem0",
        "raptor": "raptor",
    }[base_tag]
    feature_table_name = f"feat$bfi${corpus}$person_id${table_abbrev}${model_slug}${tag}"

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
        help="Minimum words per users to be considered for sampling.",
    )

    parser.add_argument(
        "--outcomes",
        nargs=5,
        metavar=tuple(t.upper() for t in TRAITS),
        default=[
            "openness_score",
            "conscientious_score",
            "extravert_score",
            "agreeable_score",
            "neurotic_score",
        ],
        help="outcome_table's trait columns.",
    )

    # Sampling for testing.
    parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample num_samples users for analysis",
    )

    parser.add_argument(
        "-n",
        type=int,
        dest="num_samples",
        default=100,
        help="Number of participants to sample.",
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
        help="OpenRouter API base URL",
    )

    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use",
    )

    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Max number of speakers processed concurrently.",
    )

    # Memory mechanism and model.
    parser.add_argument(
        "--memory",
        choices=["summary", "in_context", "batch_summary", "conversation_batch_summary", "mem0", "raptor"],
        default="summary",
        help=(
            "'summary' uses summary-mem's rolling summaries; "
            "'in_context' puts every turn in the same prompt; "
            "'batch_summary' summarizes each speaker's whole turn history in one pass; "
            "'conversation_batch_summary' summarizes each (conversation, speaker) pair in "
            "one pass, then pools those across a speaker's conversations; "
            "'mem0' uses mem0's fact-extraction memory; "
            "'raptor' uses RAPTOR's hierarchical summary tree. "
            "mem0/raptor require the optional mem0/raptor dependency groups (uv sync --group mem0 --group raptor)."
        ),
    )

    parser.add_argument(
        "--memorydb",
        default="ssubrahmanya",
        help="MySQL database to store rolling summaries.",
    )

    parser.add_argument(
        "--summary_prompt",
        default="plain",
        help="which summarization prompt SummaryMemory uses (default: %(default)s)",
    )

    parser.add_argument(
        "--max_sum_tokens",
        type=int,
        default=500,
        help="Max length (in words) of summaries produced by --memory summary/batch_summary/"
        "conversation_batch_summary (default: %(default)s).",
    )

    parser.add_argument(
        "--sum_temperature",
        type=float,
        default=0.0,
        help="Temperature passed to every --memory mechanism's summarization LLM calls; "
        "lower reduces stochasticity in summary generation (default: %(default)s).",
    )

    parser.add_argument(
        "--namespace",
        default="ds4ud",
        help="Scopes summaries in the shared MySQL table to this corpus, so they can't collide "
        "with an unrelated pipeline/corpus writing to the same --memorydb/table (default: %(default)s).",
    )

    parser.add_argument(
        "--turn_proportion",
        type=float,
        default=None,
        help="If set, keep only this share (0-1] of each speaker's turns across all of their "
        "sampled conversations, from the latest end (or earliest, with --earliest).",
    )

    parser.add_argument(
        "--earliest",
        action="store_true",
        help="With --turn_proportion, keep the earliest share of each speaker's turns (primacy). "
        "Default: keep the most recent share.",
    )

    parser.add_argument(
        "--turn_stride",
        type=int,
        default=None,
        help="If set, keep only every turn_stride-th turn per conversation, in order. "
        "Default: consume every turn (ignored if --turn_proportion is set).",
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory to store assessment results.",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:

    rng = np.random.default_rng(seed=42)
    engine = create_engine(
        f"mysql://ssubrahmanya@localhost/{args.database}?charset=utf8mb4",
        connect_args={"read_default_file": "~/.my.cnf"},
    )
    args.results_dir.mkdir(exist_ok=True)

    if args.sample:

        eligible = eligible_users(
            engine,
            args.message_table,
            args.outcome_table,
            args.user_field,
            args.conversation_field,
            args.outcomes,
            args.group_freq_thresh,
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
        users,
    )

    client = OpenAI(
        base_url=args.url,
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    # MySQL-identifier-safe tag for the feature-table/run names, kept under the
    # 64-char table-name limit.
    corpus = re.sub(r"[^a-zA-Z0-9]", "", args.namespace)[:10]

    if args.turn_proportion is not None:
        tag = f"tp{round(args.turn_proportion * 100)}{'e' if args.earliest else 'l'}"
    elif args.turn_stride is not None:
        tag = f"s{args.turn_stride}"
    else:
        tag = "all"
    if args.memory in ("summary", "batch_summary", "conversation_batch_summary"):
        # only these mechanisms' summarize() prompts read max_sum_tokens -- folding
        # it in elsewhere would just fragment in_context/mem0/raptor's naming for
        # a knob that doesn't affect them.
        tag = f"{tag}_ms{args.max_sum_tokens}"
    model_slug = re.sub(r"[^a-zA-Z0-9]", "", args.model.rsplit("/", 1)[-1])

    # folds in the turn-selection/max_sum_tokens tag and model, mirroring the
    # assessments naming convention below, so summaries from different
    # models/turn-selections/token-caps never collide in the shared MySQL table
    # (see SummaryMemory(namespace=...) etc.)
    memory_namespace = f"{args.namespace}_{tag}_{model_slug}"

    # mem0/raptor pull in heavy optional deps (mem0ai, or raptor's torch/
    # sentence-transformers stack), so they're imported lazily here rather than
    # at module level -- running any other --memory choice shouldn't require
    # either to be installed.
    def build_mem0() -> "Mem0Memory":
        from summary_mem.vendors.mem0_memory import Mem0Memory
        return Mem0Memory(client, memory_namespace, model=args.model, temperature=args.sum_temperature)

    def build_raptor() -> "RaptorMemory":
        from summary_mem.vendors.raptor_memory import RaptorMemory
        return RaptorMemory(client, memory_namespace, model=args.model, temperature=args.sum_temperature)

    # one constructor per --memory choice, dispatched on below instead of an if/elif ladder
    memory_factories = {
        "summary": lambda: SummaryMemory(
            client,
            memory_namespace,
            db_name=args.memorydb,
            model=args.model,
            summary_prompt=args.summary_prompt,
            max_sum_tokens=args.max_sum_tokens,
            temperature=args.sum_temperature,
        ),
        "batch_summary": lambda: BatchSummaryMemory(
            client,
            memory_namespace,
            db_name=args.memorydb,
            model=args.model,
            max_sum_tokens=args.max_sum_tokens,
            temperature=args.sum_temperature,
        ),
        "conversation_batch_summary": lambda: ConversationBatchSummaryMemory(
            client,
            memory_namespace,
            db_name=args.memorydb,
            model=args.model,
            max_sum_tokens=args.max_sum_tokens,
            temperature=args.sum_temperature,
        ),
        "in_context": lambda: InContextMemory(),
        "mem0": build_mem0,
        "raptor": build_raptor,
    }
    memory = memory_factories[args.memory]()

    if args.memory == "summary":
        # wipe only this run's conversations -- the MySQL store is shared across runs
        memory.db.delete_conversations([str(cid) for cid in conversation_ids])
    # else: no wipe needed -- batch_summary/conversation_batch_summary are keyed by
    # (namespace, speaker_id) or (namespace, conversation_id, speaker_id) and get
    # recomputed from this run's turns and upserted below; in_context holds no state.

    asyncio.run(
        build_memory(
            memory,
            users,
            args.message_table,
            args.conversation_field,
            args.user_field,
            args.turn_field,
            engine,
            args.max_concurrency,
            args.turn_proportion,
            args.earliest,
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
    run_tag = f"{base_tag}_{corpus}_{tag}"

    assessments = pd.DataFrame(assessments)
    store_assessments(
        assessments, 
        args.results_dir, 
        run_tag, 
        base_tag, 
        corpus, 
        model_slug, 
        tag, 
        engine
    )

    memory.close()

    outcomes = read_outcomes(
        engine,
        users,
        args.outcome_table,
        args.user_field,
        args.outcomes,
    )
    evaluate(
        assessments, 
        outcomes, 
        args.user_field, 
        args.outcomes
    )


if __name__ == "__main__":

    args = parse_args()
    main(args)