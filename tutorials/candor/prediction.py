"""CANDOR personality-prediction pipeline.

Reads turns/surveys from the `candor` MySQL database (Stage 1), stores each turn into a
``SummaryMemory`` store (Stage 2), then aggregates each speaker's context across their
sampled conversations and assesses Big Five (OCEAN) traits once per speaker, scoring
against self-reported surveys with Pearson correlation (Stage 3).

Turns come from `msgsc` (`conversation_id`, `speaker`, `turn_id`, `message`); self-reports
from `surveys` (`user_id`, `conversation_id`, `my_*`). CANDOR scores OCEAN on a ~1-5 scale
and reports **neuroticism** directly, unlike WASSA's reverse-scored "stability".

A CANDOR participant is surveyed once *per conversation*, so someone with several sampled
conversations carries several slightly different OCEAN vectors — unlike WASSA, where a
person's ``personality_*`` columns are identical across every conversation. Assessing once
per speaker from their aggregated context still matches the target better than once per
(conversation, speaker) pair; surveys are averaged per person before comparing.

``--memory`` selects the aggregation mechanism:

* ``summary``    — default; aggregate each speaker's rolling ``SummaryMemory`` summaries.
* ``in_context`` — no-external-memory baseline; feed each speaker's full turn history
  straight into the assessor.

CANDOR is large (~1,656 conversations, ~530k turns) and the summary mechanism makes one LLM
call per stored turn, so use a bounded ``--samples`` for the initial runs.

Usage:
    python prediction.py --memory summary --samples 100
    python prediction.py --memory in_context --samples 100
"""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sqlalchemy import create_engine
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import SummaryMemory

load_dotenv()

DATABASE = "candor"

RESULTS_DIR = Path(__file__).parent / "results"

# Survey columns we assess; the assessment CSV carries them as assessed_<trait>, so the
# pipeline speaks the database's own names throughout, except in the prompt itself (see
# TRAIT_NAMES) where CANDOR's abbreviations would be unfamiliar to the LLM.
TRAITS = [
    "my_open",
    "my_conscientious",
    "my_extraversion",
    "my_agreeable",
    "my_neurotic",
]

# The natural-language trait name the LLM is asked to use as its JSON key, per survey column.
TRAIT_NAMES = {
    "my_open": "openness",
    "my_conscientious": "conscientiousness",
    "my_extraversion": "extraversion",
    "my_agreeable": "agreeableness",
    "my_neurotic": "neuroticism",
}

# Shared by both mechanisms: {context} is either a distilled SummaryMemory summary
# (summary mode) or the speaker's raw turns (in_context mode) — both are a single text
# blob, so one prompt covers both.
PERSONALITY_ASSESSMENT_PROMPT = (
    "Your goal is to assess a speaker's Big Five personality traits, given a summary of "
    "their side of a conversation or the full transcript of their turns.\n\n"

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
    "present in the input — direct statements, described behavior, or patterns in language and tone. "
    "Never fabricate or infer evidence the input does not support.\n\n"

    "Respond with a single JSON object:\n"
    "{{\"openness\": <number>, \"conscientiousness\": <number>, \"extraversion\": <number>, "
    "\"agreeableness\": <number>, \"neuroticism\": <number>}}\n"
    "Always include all five keys, even when evidence for a trait is limited — give your best "
    "estimate rather than omitting the key. Output only the JSON — no markdown formatting, no "
    "surrounding text.\n\n"

    "Here is what is known about the speaker {speaker}, from a summary or transcript of their conversation:\n\n"
    "{context}\n\n"
    "Assess {speaker}'s Big Five personality trait scores (1-5) as JSON."
)


def model_slug(model: str) -> str:
    """Filesystem-safe tag for a model name, e.g. 'openai/gpt-4o-mini' -> 'gpt4omini'."""
    return re.sub(r"[^a-zA-Z0-9]", "", model.rsplit("/", 1)[-1])


def read_dataset(database: str = DATABASE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1 — read the turns and surveys from the database."""
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{database}?charset=utf8mb4")
    try:
        conversations = pd.read_sql(
            "SELECT conversation_id, speaker, turn_id, message FROM msgsc "
            "ORDER BY conversation_id, turn_id",
            engine,
        )
        surveys = pd.read_sql(
            f"SELECT user_id, conversation_id, {', '.join(TRAITS)} FROM surveys "
            "ORDER BY user_id, conversation_id",
            engine,
        )
    finally:
        engine.dispose()
    return conversations, surveys


def store_conversations(
    memory: SummaryMemory,
    conversations: pd.DataFrame,
    sample_conversation_ids: np.ndarray,
) -> None:
    """Stage 2 — store both speakers' turns into summary-mem, turn by turn."""
    for conversation_id in tqdm(sample_conversation_ids, desc="storing conversations"):

        turns = conversations[conversations["conversation_id"] == conversation_id]
        turns = turns.sort_values("turn_id")

        for _, turn in turns.iterrows():
            memory.update(
                conversation_id=str(turn["conversation_id"]),
                speaker_id=str(turn["speaker"]),
                turn_text=turn["message"],
            )


def build_speaker_context(
    speaker_id: str,
    memory: SummaryMemory | None,
    conversations: pd.DataFrame,
    sample_conversation_ids: np.ndarray,
) -> str:
    """Aggregate one speaker's context across every one of their sampled conversations.

    Summary mode (``memory`` is a ``SummaryMemory``) recalls its rolling summaries for
    ``speaker_id``; in_context mode falls back to that speaker's full turn history,
    read from ``conversations``. Either way, the per-conversation pieces are joined
    into the single blob ``PERSONALITY_ASSESSMENT_PROMPT`` expects.
    """
    if isinstance(memory, SummaryMemory):
        contexts = memory.recall_all(speaker_id)
    else:
        turns = conversations[
            conversations["conversation_id"].isin(sample_conversation_ids)
            & (conversations["speaker"].astype(str) == speaker_id)
        ]
        contexts = [
            "\n".join(group.sort_values("turn_id")["message"].astype(str))
            for _, group in turns.groupby("conversation_id")
        ]

    context = "\n\n".join(f"Conversation {i + 1}:\n{c}" for i, c in enumerate(contexts))
    return context


def assess_personality(
    chat_client: OpenAI,
    model: str,
    speaker: str,
    context: str,
) -> dict[str, float]:
    response = chat_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PERSONALITY_ASSESSMENT_PROMPT.format(
                    speaker=speaker,
                    context=context,
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    return {trait: float(scores[TRAIT_NAMES[trait]]) for trait in TRAITS}


def evaluate(assessments: pd.DataFrame, surveys: pd.DataFrame) -> None:
    """Score assessments against self-reported surveys (Pearson r per trait).

    Surveys are averaged per person (``user_id``) before comparing — a participant
    is surveyed once per conversation and can carry slightly different OCEAN vectors
    across conversations.
    """
    person_surveys = surveys.groupby("user_id", as_index=False)[TRAITS].mean()
    comparison = pd.merge(
        assessments,
        person_surveys,
        left_on="speaker_id",
        right_on="user_id",
        how="left",
    )

    for trait in TRAITS:
        valid = comparison[[f"assessed_{trait}", trait]].dropna()
        if len(valid) < 2:
            print(f"{trait:18s} Pearson r = n/a  (n={len(valid)}; need >=2 points)")
            continue
        r, p = pearsonr(valid[f"assessed_{trait}"], valid[trait])
        print(f"{trait:18s} Pearson r = {r:.3f}  (n={len(valid)})  p-value = {p:.3e}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        default=DATABASE,
        help="MySQL database holding the CANDOR corpus (default: %(default)s).",
    )
    parser.add_argument(
        "--memory",
        choices=["summary", "in_context"],
        default="summary",
        help=(
            "Recall mechanism: 'summary' uses summary-mem's rolling summaries; "
            "'in_context' is the no-external-memory baseline that puts every turn "
            "in the same prompt."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default="candor_memory.db",
        help="Path to the SummaryMemory SQLite store (recreated fresh on each run).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of conversations to sample and run the pipeline over.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use (default: %(default)s to openai/gpt-4o-mini).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    rng = np.random.default_rng()

    conversations, surveys = read_dataset(args.database)

    conversation_ids = conversations["conversation_id"].unique()
    sample_conversation_ids = rng.choice(
        conversation_ids,
        size=min(args.samples, len(conversation_ids)),
        replace=False,
    )
    speakers = sorted(
        conversations.loc[
            conversations["conversation_id"].isin(sample_conversation_ids), "speaker"
        ]
        .astype(str)
        .unique()
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    
    memory = None
    if args.memory == "summary":

        if args.db.exists():
            args.db.unlink()  # fresh store so re-running starts from an empty memory

        memory = SummaryMemory(client, db_path=args.db, model=args.model)
        store_conversations(memory, conversations, sample_conversation_ids)

    assessments = []
    for speaker in tqdm(speakers, desc="assessing personality traits"):

        # for every speaker, aggregate their context across all sampled conversations
        context = build_speaker_context(
            speaker, 
            memory, 
            conversations, 
            sample_conversation_ids
        )

        # then assess their Big Five traits from that context
        assessment = assess_personality(
            client,
            args.model,
            speaker,
            context,
        )

        assessments.append(
            {
                "speaker_id": speaker,
                **{f"assessed_{trait}": assessment[trait] for trait in TRAITS},
            }
        )

    assessments = pd.DataFrame(assessments)
    RESULTS_DIR.mkdir(exist_ok=True)
    assessments.to_csv(
        RESULTS_DIR / f"assessments_{args.memory}_{model_slug(args.model)}.csv", index=False
    )
    if memory is not None:
        memory.close()

    evaluate(assessments, surveys)