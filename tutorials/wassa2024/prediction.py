"""WASSA 2024 personality-prediction pipeline.

Reads conversation/survey data (Stage 1), stores each turn into a ``SummaryMemory`` store
(Stage 2), then aggregates each speaker's context across their sampled conversations and
assesses Big Five (OCEAN) traits once per speaker, scoring against self-reported surveys
with Pearson correlation — the PER track's metric (Stage 3).

Personality is a trait of the person, not of a conversation: a person's ``personality_*``
survey columns are identical across every conversation they took part in. So assessing
once per speaker, from their aggregated context, matches the target better than once per
(conversation, speaker) pair.

``--memory`` selects the aggregation mechanism:

* ``summary``    — default; aggregate each speaker's rolling ``SummaryMemory`` summaries.
* ``in_context`` — no-external-memory baseline; feed each speaker's full turn history
  straight into the assessor.

Usage:
    python prediction.py --dataset ../../data/wassa2024/train --memory summary
    python prediction.py --dataset ../../data/wassa2024/train --memory in_context
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import SummaryMemory

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# Maps a trait to (survey column suffix, invert-before-comparing) where it differs from
# the assessed name. Neuroticism is the inverse of the survey's "stability" self-report.
SURVEY_TRAIT_OVERRIDES = {"neuroticism": ("stability", True)}

# Shared by both mechanisms: {context} is either a list of rolling SummaryMemory summaries
# (summary mode) or the speaker's raw turns (in_context mode) — build_speaker_context()
# produces the same shape either way.
PERSONALITY_ASSESSMENT_PROMPT = (
    "Your goal is to assess a speaker's Big Five personality traits, "
    "given a list of their conversational turns or the summaries of the turns.\n\n"

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
    "Score each trait on a continuous 1-7 scale (1 = very low, 4 = moderate, 7 = very high), based only on evidence "
    "present in the input — direct statements, described behavior, or patterns in language and tone. "
    "Never fabricate or infer evidence the input does not support.\n\n"

    "Respond with a single JSON object:\n"
    "{{\"openness\": <number>, \"conscientiousness\": <number>, \"extraversion\": <number>, "
    "\"agreeableness\": <number>, \"neuroticism\": <number>}}\n"
    "Always include all five keys, even when evidence for a trait is limited — give your best "
    "estimate rather than omitting the key. Output only the JSON — no markdown formatting, no "
    "surrounding text.\n\n"

    "Here is what is known about {speaker}, from a list of summaries or conversational turns:\n\n"
    "{context}\n\n"
    "Assess {speaker}'s Big Five personality trait scores (1-7) as JSON."
)


def model_slug(model: str) -> str:
    """Filesystem-safe tag for a model name, e.g. 'openai/gpt-4o-mini' -> 'gpt4omini'."""
    return re.sub(r"[^a-zA-Z0-9]", "", model.rsplit("/", 1)[-1])


def read_dataset(datafolder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1 — read the conversation and survey data."""
    conversations = pd.read_csv(datafolder / "conversations.csv")
    surveys = pd.read_csv(datafolder / "surveys.csv")
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
                speaker_id=str(turn["speaker_id"]),
                turn_text=turn["text"],
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
            & (conversations["speaker_id"].astype(str) == speaker_id)
        ]
        contexts = [
            "\n".join(group.sort_values("turn_id")["text"].astype(str))
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
    return {trait: float(scores[trait]) for trait in TRAITS}


def evaluate(assessments: pd.DataFrame, surveys: pd.DataFrame) -> None:
    """Score assessments against self-reported surveys (Pearson r per trait).

    Surveys are deduplicated to one row per person — ``personality_*`` columns are
    identical across every row a person appears in — before comparing against the
    per-speaker assessments.
    """
    person_surveys = surveys.drop_duplicates("person_id")
    comparison = pd.merge(
        assessments,
        person_surveys,
        left_on="speaker_id",
        right_on="person_id",
        how="left",
    )

    for trait in TRAITS:
        survey_trait, invert = SURVEY_TRAIT_OVERRIDES.get(trait, (trait, False))
        valid = comparison[[f"assessed_{trait}", f"personality_{survey_trait}"]].dropna()
        if len(valid) < 2:
            print(f"{trait:18s} Pearson r = n/a  (n={len(valid)}; need >=2 points)")
            continue
        target = valid[f"personality_{survey_trait}"]
        if invert:
            target = 8 - target  # 1-7 scale: flip stability into neuroticism
        r, p = pearsonr(valid[f"assessed_{trait}"], target)
        print(f"{trait:18s} Pearson r = {r:.3f}  (n={len(valid)})  p-value = {p:.3e}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset split folder (e.g. ../../data/wassa2024/train).",
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
        default="wassa2024_memory.db",
        help="Path to the SummaryMemory SQLite store (recreated fresh on each run).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of conversations to sample and run the pipeline over.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling conversations.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LLM model to use (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    conversations, surveys = read_dataset(args.dataset)

    conversation_ids = conversations["conversation_id"].unique()
    sample_conversation_ids = rng.choice(
        conversation_ids,
        size=min(args.samples, len(conversation_ids)),
        replace=False,
    )
    speakers = sorted(
        conversations.loc[
            conversations["conversation_id"].isin(sample_conversation_ids), "speaker_id"
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

        context = build_speaker_context(speaker, memory, conversations, sample_conversation_ids)
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