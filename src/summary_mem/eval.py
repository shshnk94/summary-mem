"""Evaluate the running-summary memory on LongMemEval-style data.

This is the *evaluation* counterpart to memLLM's ``generate.py``, but shaped
like the HippoRAG quick start (``index`` -> ``rag_qa``) and specialized for
long-term-memory recall:

1. **index** a multi-session conversation ("haystack") into the memory, exactly
   as it would accumulate in a real chat — turn by turn, session by session;
2. **rag_qa** a question whose answer lives somewhere in those sessions,
   answering *only* from the recalled running summary;
3. **judge** the answer against the gold answer with an LLM, the way
   LongMemEval scores correctness.

Because the mechanism keeps a single rolling summary per speaker (no per-turn
retrieval), this measures the thing that actually matters for a summary memory:
**does the needle survive summarization across many sessions?**

Quick start (runs a built-in toy haystack, no dataset needed)::

    python -m summary_mem.eval

Run a real LongMemEval file::

    python -m summary_mem.eval --data longmemeval_s.json --limit 50

LongMemEval instance format (https://github.com/xiaowu0162/LongMemEval)::

    {
      "question_id": "...",
      "question_type": "multi-session",
      "question": "...",
      "answer": "...",
      "question_date": "2023/05/20",
      "haystack_dates": ["2023/04/01", ...],
      "haystack_sessions": [[{"role": "user", "content": "..."},
                             {"role": "assistant", "content": "..."}], ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from .memory import SummaryMemory
from .prompts.qa import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPLATE,
    QA_SYSTEM_PROMPT,
    QA_TEMPLATE,
)

load_dotenv()


def get_chat_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    """OpenAI client for chat/summarization calls (local vLLM server by default)."""
    return OpenAI(
        base_url=base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@dataclass
class QAResult:
    """Outcome of answering (and optionally grading) one question."""

    question: str
    answer: str
    memory: dict[str, str]          # recalled {speaker: running summary}
    gold: str | None = None
    correct: bool | None = None     # None when no gold answer was supplied


class MemoryEvaluator:
    """Index sessions into a :class:`SummaryMemory`, then answer/grade questions.

    One evaluator owns one conversation's worth of memory; build a fresh one per
    LongMemEval instance so haystacks never leak into each other (see
    :func:`evaluate_dataset`).
    """

    def __init__(
        self,
        chat_client: OpenAI,
        *,
        conversation_id: str = "eval",
        model: str = "Qwen/Qwen3-8B",
        memory: SummaryMemory | None = None,
        db_name: str = "ssubrahmanya",
        namespace: str = "longmemeval",
    ) -> None:
        self.chat_client = chat_client
        self.conversation_id = conversation_id
        self.model = model
        self.memory = memory or SummaryMemory(chat_client, model=model, db_name=db_name, namespace=namespace)

    # -- indexing -----------------------------------------------------------

    def index(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        """Store a haystack of sessions into memory, oldest first.

        ``sessions`` is a list of sessions; each session is a list of turns
        ``{"role"|"speaker": ..., "content"|"text": ...}`` (the LongMemEval
        shape). Turns are folded into the running summary **batched per session
        per speaker** — one ``memory.update`` call per speaker per session —
        which keeps the incremental, session-granular update behavior without
        an LLM call for every single turn (haystacks can be hundreds of turns).

        ``dates`` (optional, parallel to ``sessions``) is prepended to each
        batch so temporal-reasoning questions have a timestamp to work from.
        """
        for i, session in enumerate(sessions):
            date = dates[i] if dates and i < len(dates) else None
            # Group consecutive content by speaker, preserving order.
            by_speaker: dict[str, list[str]] = {}
            order: list[str] = []
            for turn in session:
                speaker = str(turn.get("role") or turn.get("speaker") or "user")
                text = str(turn.get("content") or turn.get("text") or "").strip()
                if not text:
                    continue
                if speaker not in by_speaker:
                    by_speaker[speaker] = []
                    order.append(speaker)
                by_speaker[speaker].append(text)
            for speaker in order:
                blob = "\n".join(by_speaker[speaker])
                if date:
                    blob = f"[{date}]\n{blob}"
                self.memory.update(self.conversation_id, speaker, blob)

    # -- question answering -------------------------------------------------

    def rag_qa(
        self,
        question: str,
        *,
        question_date: str | None = None,
        gold_answer: str | None = None,
    ) -> QAResult:
        """Answer ``question`` from the recalled memory; grade it if gold given."""
        summaries = self.memory.recall(self.conversation_id)
        answer = self._answer(question, summaries, question_date)
        correct = None
        if gold_answer is not None:
            correct = self.judge(question, gold_answer, answer)
        return QAResult(
            question=question,
            answer=answer,
            memory=summaries,
            gold=gold_answer,
            correct=correct,
        )

    def judge(self, question: str, gold: str, hypothesis: str) -> bool:
        """LLM-as-judge: is ``hypothesis`` a correct answer given ``gold``?"""
        prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, hypothesis=hypothesis)
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=8,
        )
        verdict = (response.choices[0].message.content or "").strip().lower()
        return verdict.startswith("yes")

    def close(self) -> None:
        self.memory.close()

    # -- internals ----------------------------------------------------------

    def _answer(self, question: str, summaries: dict[str, str], question_date: str | None) -> str:
        memory_block = (
            "\n\n".join(f"{speaker}:\n{summary}" for speaker, summary in summaries.items())
            or "(no memory available)"
        )
        prompt = QA_TEMPLATE.format(
            question_date=question_date or "Unknown",
            memory=memory_block,
            question=question,
        )
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()


def evaluate_dataset(
    path: Path,
    *,
    chat_client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-8B",
    limit: int | None = None,
    db_name: str = "ssubrahmanya",
    namespace: str = "longmemeval",
) -> dict:
    """Run the full LongMemEval-format dataset and report accuracy.

    A fresh memory is built per instance so haystacks stay isolated (mirroring
    ``generate.py``'s one-memory-per-conversation rule). Returns a summary dict
    with overall accuracy and a per-question-type breakdown.
    """
    chat_client = chat_client or get_chat_client()
    instances = json.loads(Path(path).read_text())
    if limit is not None:
        instances = instances[:limit]

    total = Counter()
    correct = Counter()
    records = []

    for inst in tqdm.tqdm(instances, desc=Path(path).name):
        qtype = inst.get("question_type", "unknown")
        evaluator = MemoryEvaluator(
            chat_client,
            conversation_id=inst.get("question_id", "eval"),
            model=model,
            db_name=db_name,
            namespace=namespace,
        )
        try:
            evaluator.index(inst["haystack_sessions"], inst.get("haystack_dates"))
            result = evaluator.rag_qa(
                inst["question"],
                question_date=inst.get("question_date"),
                gold_answer=inst.get("answer"),
            )
        finally:
            evaluator.close()

        total[qtype] += 1
        total["__all__"] += 1
        if result.correct:
            correct[qtype] += 1
            correct["__all__"] += 1
        records.append(
            {
                "question_id": inst.get("question_id"),
                "question_type": qtype,
                "question": result.question,
                "gold": result.gold,
                "answer": result.answer,
                "correct": result.correct,
            }
        )

    def acc(key: str) -> float:
        return correct[key] / total[key] if total[key] else 0.0

    by_type = {qt: {"n": total[qt], "accuracy": acc(qt)} for qt in sorted(total) if qt != "__all__"}
    return {
        "n": total["__all__"],
        "accuracy": acc("__all__"),
        "by_question_type": by_type,
        "records": records,
    }


# A tiny built-in haystack so `python -m summary_mem.eval` works with no data:
# the answer (a dog named "Marbles") is mentioned in one early session and
# must survive being summarized across the later, unrelated sessions.
_TOY_SESSIONS = [
    [
        {"role": "user", "content": "I just adopted a border collie puppy named Marbles."},
        {"role": "assistant", "content": "Congrats! Border collies are wonderfully energetic."},
    ],
    [
        {"role": "user", "content": "Work has been hectic — we shipped the billing revamp this week."},
        {"role": "assistant", "content": "Nice, shipping a billing revamp is a big deal."},
    ],
    [
        {"role": "user", "content": "I'm training for a half marathon in the fall."},
        {"role": "assistant", "content": "Great goal! Consistent long runs will help."},
    ],
]
_TOY_QUESTION = "What is the name of my dog, and what breed is it?"
_TOY_ANSWER = "A border collie named Marbles."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate summary-mem's recall on LongMemEval-style data.",
    )
    parser.add_argument("--data", type=Path, default=None, help="LongMemEval JSON file (omit for the built-in toy run)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="LLM for answering, summarizing, and judging")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N instances")
    parser.add_argument("--out", type=Path, default=None, help="write per-question records as JSON here")
    parser.add_argument("--db-name", default="ssubrahmanya", help="MySQL database to store summaries in")
    parser.add_argument(
        "--namespace",
        default="longmemeval",
        help="Scopes summaries in the shared MySQL table to this eval run, so they can't collide "
        "with an unrelated pipeline/corpus writing to the same db_name/table (default: %(default)s).",
    )
    args = parser.parse_args()

    chat_client = get_chat_client()

    if args.data is None:
        # Quick start: index the toy haystack, ask, grade.
        ev = MemoryEvaluator(
            chat_client,
            conversation_id="toy",
            model=args.model,
            db_name=args.db_name,
            namespace=args.namespace,
        )
        try:
            ev.index(_TOY_SESSIONS)
            result = ev.rag_qa(_TOY_QUESTION, gold_answer=_TOY_ANSWER)
        finally:
            ev.close()
        print("=== recalled memory ===")
        for speaker, summary in result.memory.items():
            print(f"[{speaker}] {summary}")
        print(f"\nQ: {result.question}")
        print(f"A: {result.answer}")
        print(f"gold: {result.gold}")
        print(f"correct: {result.correct}")
        return

    report = evaluate_dataset(
        args.data,
        chat_client=chat_client,
        model=args.model,
        limit=args.limit,
        db_name=args.db_name,
        namespace=args.namespace,
    )
    print(f"\noverall accuracy: {report['accuracy']:.3f}  (n={report['n']})")
    for qtype, stats in report["by_question_type"].items():
        print(f"  {qtype:28s} {stats['accuracy']:.3f}  (n={stats['n']})")
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote per-question records to {args.out}")


if __name__ == "__main__":
    main()
