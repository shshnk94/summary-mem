"""WASSA 2024 personality-prediction pipeline.

Runs the three stages from the "Experiments" section of ``wassa2024.ipynb`` end to end:

1. Read the conversation and survey data for a dataset split.
2. Store both speakers' turns, turn by turn, into a ``SummaryMemory`` store.
3. Recall each speaker's rolling summary and predict their Big Five (OCEAN)
   personality traits, then score the predictions against the self-reported
   surveys with Pearson correlation (the PER track's official metric).

Usage:
    python prediction.py --dataset ../../data/wassa2024/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from summary_mem import SummaryMemory
from summary_mem.clients import get_chat_client

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "stability",
]

SYSTEM_PROMPT = (
    "Given a factual summary of a single person built from their side of a conversation, "
    "estimate their Big Five (OCEAN) personality traits: "
    "openness, conscientiousness, extraversion, agreeableness, and emotional stability. "
    "Score each trait on a continuous 1-7 scale (1 = very low, 7 = very high). "
    "Reply with ONLY a JSON object whose keys are exactly: "
    "openness, conscientiousness, extraversion, agreeableness, stability."
)

PERSONALITY_PREDICTION_TEMPLATE = (
    "Here is what is known about the speaker {speaker}, summarized from their conversation:\n\n{context}\n\n"
    "Estimate {speaker}'s Big Five personality trait scores (1-7) as JSON."
)


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

        # read all turns for this conversation, sorted by turn_id
        turns = conversations[conversations["conversation_id"] == conversation_id]
        turns = turns.sort_values("turn_id")

        for _, turn in turns.iterrows():
            memory.update(
                conversation_id=str(turn["conversation_id"]),
                speaker_id=str(turn["speaker_id"]),
                turn_text=turn["text"],
            )


def predict_personality(memory: SummaryMemory, speaker: str, context: str) -> dict[str, float]:
    response = memory.chat_client.chat.completions.create(
        model=memory.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PERSONALITY_PREDICTION_TEMPLATE.format(
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


def predict(
    memory: SummaryMemory,
    sample_conversation_ids: np.ndarray,
) -> pd.DataFrame:
    """Stage 3 — recall each speaker's summary and predict their OCEAN traits."""
    predictions = []
    for conversation_id in tqdm(sample_conversation_ids, desc="predicting personality traits"):

        recalled = memory.recall(str(conversation_id))
        for speaker, summary in recalled.items():

            pred = predict_personality(memory, speaker, summary)
            predictions.append(
                {
                    "conversation_id": conversation_id,
                    "speaker_id": speaker,
                    **{f"pred_{trait}": pred[trait] for trait in TRAITS},
                }
            )

    return pd.DataFrame(predictions)


def evaluate(predictions: pd.DataFrame, surveys: pd.DataFrame) -> None:
    """Score predictions against self-reported surveys (Pearson r per trait)."""
    comparison = pd.merge(
        predictions,
        surveys,
        left_on=["conversation_id", "speaker_id"],
        right_on=["conversation_id", "person_id"],
        how="left",
    )

    for trait in TRAITS:
        valid = comparison[[f"pred_{trait}", f"personality_{trait}"]].dropna()
        if len(valid) < 2:
            print(f"{trait:18s} Pearson r = n/a  (n={len(valid)}; need >=2 points)")
            continue
        r, p = pearsonr(valid[f"pred_{trait}"], valid[f"personality_{trait}"])
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
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Stage 1 — read the dataset.
    conversations, surveys = read_dataset(args.dataset)

    # Fresh store so re-running starts from an empty memory.
    if args.db.exists():
        args.db.unlink()

    client = get_chat_client()
    memory = SummaryMemory(client, db_path=args.db)

    conversation_ids = conversations["conversation_id"].unique()
    sample_conversation_ids = rng.choice(
        conversation_ids,
        size=min(args.samples, len(conversation_ids)),
        replace=False,
    )

    # Stage 2 — store the sampled conversations into memory.
    store_conversations(memory, conversations, sample_conversation_ids)

    # Stage 3 — predict personality and score it.
    predictions = predict(memory, sample_conversation_ids)
    predictions.to_csv("predictions.csv", index=False)
    memory.close()

    evaluate(predictions, surveys)    