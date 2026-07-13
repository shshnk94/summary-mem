"""CANDOR personality-prediction pipeline.

Runs the three stages from the "Experiments" section of ``candor.ipynb`` end to end:

1. Read the turns and surveys from the `candor` MySQL database.
2. Store both speakers' turns, turn by turn, into a ``SummaryMemory`` store.
3. Recall each speaker's rolling summary and assess their Big Five (OCEAN) personality
   traits, then score the assessments against the self-reported surveys with Pearson
   correlation.

The corpus is read from the database under its own column names: turns come from `msgsc`
(`conversation_id`, `speaker`, `turn_id`, `message`) and the self-reports from `surveys`
(`user_id`, `conversation_id`, `my_*`). CANDOR scores OCEAN on a ~1-5 scale and reports
**neuroticism** directly, unlike WASSA's reverse-scored "stability".

The recall step supports two mechanisms, selected with ``--memory``:

* ``summary``    — the default; recall each speaker's rolling ``SummaryMemory`` summary.
* ``in_context`` — the no-external-memory baseline; feed each speaker's full turn history
  straight into the assessor, so every turn is present in the same prompt.

CANDOR is large (~1,656 conversations, ~530k turns) and the summary mechanism makes one LLM
call per stored turn, so use a bounded ``--samples`` for the initial runs.

Usage:
    python prediction.py --memory summary --samples 100
    python prediction.py --memory in_context --samples 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sqlalchemy import create_engine
from tqdm.auto import tqdm

from openai import OpenAI

from summary_mem import SummaryMemory
from summary_mem.clients import get_chat_client
from summary_mem.config import DEFAULT_LLM_MODEL

DATABASE = "candor"

# The survey columns we assess. The LLM is asked for these keys verbatim, and the assessment
# CSV carries them as assessed_<trait>, so the whole pipeline speaks the database's own names.
TRAITS = [
    "my_open",
    "my_conscientious",
    "my_extraversion",
    "my_agreeable",
    "my_neurotic",
]

# What each survey column actually measures — spelled out for the LLM, which cannot be
# expected to know CANDOR's abbreviations.
TRAIT_DESCRIPTIONS = {
    "my_open": "openness to experience",
    "my_conscientious": "conscientiousness",
    "my_extraversion": "extraversion",
    "my_agreeable": "agreeableness",
    "my_neurotic": "neuroticism",
}

# One prompt per recall mechanism. The instruction and the framing of {context} differ:
# 'summary' presents a distilled summary, whereas 'in_context' presents the raw turns.
PERSONALITY_ASSESSMENT_PROMPTS = {
    "summary": (
        "Given a factual summary of a single person built from their side of a conversation, "
        "assess their Big Five (OCEAN) personality traits. "
        "Score each trait on a continuous 1-5 scale (1 = very low, 5 = very high). "
        "Reply with ONLY a JSON object whose keys are exactly:\n{traits}\n\n"
        "Here is what is known about the speaker {speaker}, summarized from their conversation:\n\n{context}\n\n"
        "Assess {speaker}'s Big Five personality trait scores (1-5) as JSON."
    ),
    "in_context": (
        "Given the full transcript of a single person's turns from a conversation, "
        "assess their Big Five (OCEAN) personality traits. "
        "Score each trait on a continuous 1-5 scale (1 = very low, 5 = very high). "
        "Reply with ONLY a JSON object whose keys are exactly:\n{traits}\n\n"
        "Here are all of the turns spoken by {speaker} in the conversation, in order:\n\n{context}\n\n"
        "Assess {speaker}'s Big Five personality trait scores (1-5) as JSON."
    ),
}

# The key list handed to the model, each key annotated with what it measures.
TRAIT_KEYS = "\n".join(f"- {trait} ({TRAIT_DESCRIPTIONS[trait]})" for trait in TRAITS)


def read_dataset(database: str = DATABASE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1 — read the turns and surveys from the database."""
    engine = create_engine(f"mysql://ssubrahmanya@localhost/{database}?charset=utf8mb4")
    try:
        conversations = pd.read_sql(
            "SELECT conversation_id, speaker, turn_id, message FROM msgsc", engine
        )
        surveys = pd.read_sql(
            f"SELECT user_id, conversation_id, {', '.join(TRAITS)} FROM surveys", engine
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

        # read all turns for this conversation, sorted by turn_id
        turns = conversations[conversations["conversation_id"] == conversation_id]
        turns = turns.sort_values("turn_id")

        for _, turn in turns.iterrows():
            memory.update(
                conversation_id=str(turn["conversation_id"]),
                speaker_id=str(turn["speaker"]),
                turn_text=turn["message"],
            )


def gather_turns(conversations: pd.DataFrame, conversation_id) -> dict[str, str]:
    """In-context alternative to ``SummaryMemory.recall``.

    Instead of a summary, return each speaker's full turn history verbatim (turns joined in
    order) so the assessor sees every turn in the same prompt. The return shape matches
    ``recall`` — ``{speaker: context}`` — so ``assess`` is agnostic to which mechanism
    produced the context.
    """
    turns = conversations[conversations["conversation_id"] == conversation_id]
    turns = turns.sort_values("turn_id")

    contexts: dict[str, list[str]] = {}
    for _, turn in turns.iterrows():
        speaker = str(turn["speaker"])
        contexts.setdefault(speaker, []).append(str(turn["message"]))

    return {speaker: "\n".join(texts) for speaker, texts in contexts.items()}


def assess_personality(
    chat_client: OpenAI,
    model: str,
    speaker: str,
    context: str,
    memory_mode: str,
) -> dict[str, float]:
    prompt = PERSONALITY_ASSESSMENT_PROMPTS[memory_mode]
    response = chat_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt.format(
                    traits=TRAIT_KEYS,
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


def assess(
    chat_client: OpenAI,
    model: str,
    sample_conversation_ids: np.ndarray,
    recall_fn,
    memory_mode: str,
) -> pd.DataFrame:

    """Stage 3 — gather each speaker's context and assess their OCEAN traits.

    ``recall_fn(conversation_id) -> {speaker: context}`` supplies the per-speaker context,
    which is either a rolling summary (``summary`` mode) or the full turn history
    (``in_context`` mode). ``memory_mode`` also selects the matching prompt.
    """
    assessments = []
    for conversation_id in tqdm(sample_conversation_ids, desc="assessing personality traits"):

        recalled = recall_fn(conversation_id)
        for speaker, context in recalled.items():

            assessment = assess_personality(
                chat_client,
                model,
                speaker,
                context,
                memory_mode,
            )

            assessments.append(
                {
                    "conversation_id": conversation_id,
                    "speaker": speaker,
                    **{f"assessed_{trait}": assessment[trait] for trait in TRAITS},
                }
            )

    return pd.DataFrame(assessments)


def evaluate(assessments: pd.DataFrame, surveys: pd.DataFrame) -> None:
    """Score assessments against self-reported surveys (Pearson r per trait)."""
    comparison = pd.merge(
        assessments,
        surveys,
        left_on=["conversation_id", "speaker"],
        right_on=["conversation_id", "user_id"],
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling conversations.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Stage 1 — read the dataset from the database.
    conversations, surveys = read_dataset(args.database)

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
        # no external memory: gather each speaker's full turn history at assessment time.
        recall_fn = lambda conversation_id: gather_turns(conversations, conversation_id)

    model = memory.model if memory is not None else DEFAULT_LLM_MODEL

    # Stage 3 — assess personality and score it.
    assessments = assess(
        client,
        model,
        sample_conversation_ids,
        recall_fn,
        args.memory,
    )

    assessments.to_csv(f"assessments_{args.memory}.csv", index=False)
    if memory is not None:
        memory.close()

    evaluate(assessments, surveys)
