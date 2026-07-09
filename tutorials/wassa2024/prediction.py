"""WASSA 2024 personality-prediction pipeline.

Runs the three stages from the "Experiments" section of ``wassa2024.ipynb`` end to end:

1. Read the conversation and survey data for a dataset split.
2. Store both speakers' turns, turn by turn, into a ``SummaryMemory`` store.
3. Recall each speaker's rolling summary and predict their Big Five (OCEAN)
   personality traits, then score the predictions against the self-reported
   surveys with Pearson correlation (the PER track's official metric).

The recall step supports two mechanisms, selected with ``--memory``:

* ``summary``    — the default; recall each speaker's rolling ``SummaryMemory`` summary.
* ``in_context`` — the no-external-memory baseline; feed each speaker's full turn
  history straight into the predictor, so every turn is present in the same prompt.

Usage:
    python prediction.py --dataset ../../data/wassa2024/train --memory summary
    python prediction.py --dataset ../../data/wassa2024/train --memory in_context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import SummaryMemory
from summary_mem.clients import get_chat_client
from summary_mem.config import DEFAULT_LLM_MODEL

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "stability",
]

# One prompt per recall mechanism. The instruction and the framing of {context} differ:
# 'summary' presents a distilled summary, whereas 'in_context' presents the raw turns.
PERSONALITY_PREDICTION_PROMPTS = {
    "summary": (
        "Given a factual summary of a single person built from their side of a conversation, "
        "estimate their Big Five (OCEAN) personality traits: "
        "openness, conscientiousness, extraversion, agreeableness, and emotional stability. "
        "Score each trait on a continuous 1-7 scale (1 = very low, 7 = very high). "
        "Reply with ONLY a JSON object whose keys are exactly: "
        "openness, conscientiousness, extraversion, agreeableness, stability.\n\n"
        "Here is what is known about the speaker {speaker}, summarized from their conversation:\n\n{context}\n\n"
        "Estimate {speaker}'s Big Five personality trait scores (1-7) as JSON."
    ),
    "in_context": (
        "Given the full transcript of a single person's turns from a conversation, "
        "estimate their Big Five (OCEAN) personality traits: "
        "openness, conscientiousness, extraversion, agreeableness, and emotional stability. "
        "Score each trait on a continuous 1-7 scale (1 = very low, 7 = very high). "
        "Reply with ONLY a JSON object whose keys are exactly: "
        "openness, conscientiousness, extraversion, agreeableness, stability.\n\n"
        "Here are all of the turns spoken by {speaker} in the conversation, in order:\n\n{context}\n\n"
        "Estimate {speaker}'s Big Five personality trait scores (1-7) as JSON."
    ),
}


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


def gather_turns(conversations: pd.DataFrame, conversation_id) -> dict[str, str]:
    """In-context alternative to ``SummaryMemory.recall``.

    Instead of a summary, return each speaker's full turn history verbatim (turns
    joined in order) so the predictor sees every turn in the same prompt. The return
    shape matches ``recall`` — ``{speaker_id: context}`` — so ``predict`` is agnostic
    to which mechanism produced the context.
    """
    turns = conversations[conversations["conversation_id"] == conversation_id]
    turns = turns.sort_values("turn_id")

    contexts: dict[str, list[str]] = {}
    for _, turn in turns.iterrows():
        speaker = str(turn["speaker_id"])
        contexts.setdefault(speaker, []).append(str(turn["text"]))

    return {speaker: "\n".join(texts) for speaker, texts in contexts.items()}


def predict_personality(
    chat_client: OpenAI,
    model: str,
    speaker: str,
    context: str,
    memory_mode: str,
) -> dict[str, float]:
    prompt = PERSONALITY_PREDICTION_PROMPTS[memory_mode]
    response = chat_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt.format(
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
    chat_client: OpenAI,
    model: str,
    sample_conversation_ids: np.ndarray,
    recall_fn,
    memory_mode: str,
) -> pd.DataFrame:
    
    """Stage 3 — gather each speaker's context and predict their OCEAN traits.

    ``recall_fn(conversation_id) -> {speaker_id: context}`` supplies the per-speaker
    context, which is either a rolling summary (``summary`` mode) or the full turn
    history (``in_context`` mode). ``memory_mode`` also selects the matching prompt.
    """
    predictions = []
    for conversation_id in tqdm(sample_conversation_ids, desc="predicting personality traits"):

        recalled = recall_fn(conversation_id)
        for speaker, context in recalled.items():

            pred = predict_personality(
                chat_client,
                model,
                speaker,
                context,
                memory_mode,
            )
            
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
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Stage 1 — read the dataset.
    conversations, surveys = read_dataset(args.dataset)

    conversation_ids = conversations["conversation_id"].unique()
    sample_conversation_ids = rng.choice(
        conversation_ids,
        size=min(args.samples, len(conversation_ids)),
        replace=False,
    )

    client = get_chat_client()
    memory = None

    # Stage 2 — build the per-speaker context for the selected mechanism.
    if args.memory == "summary":

        # Fresh store so re-running starts from an empty memory.
        if args.db.exists():
            args.db.unlink()

        # SummaryMemory is only needed when we actually summarize.
        memory = SummaryMemory(client, db_path=args.db)

        # summary-mem: store turns so a rolling summary is built, then recall it.
        store_conversations(memory, conversations, sample_conversation_ids)
        recall_fn = lambda conversation_id: memory.recall(str(conversation_id))
    else:  # in_context
        # no external memory: gather each speaker's full turn history at predict time.
        recall_fn = lambda conversation_id: gather_turns(conversations, conversation_id)

    model = memory.model if memory is not None else DEFAULT_LLM_MODEL

    # Stage 3 — predict personality and score it.
    predictions = predict(
        client,
        model,
        sample_conversation_ids,
        recall_fn,
        args.memory,
    )

    predictions.to_csv(f"predictions_{args.memory}.csv", index=False)
    if memory is not None:
        memory.close()

    evaluate(predictions, surveys)    