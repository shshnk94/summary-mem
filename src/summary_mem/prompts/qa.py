"""Prompts for memory-grounded QA and LLM-judge scoring.

Used by ``summary_mem.eval`` to (a) answer a question from the recalled
running summaries and (b) grade that answer against a gold answer, in the
style of LongMemEval's GPT-4o judge.
"""

QA_SYSTEM_PROMPT = (
    "You answer questions using only your memory of a long, multi-session "
    "conversation with the user. The memory is a running summary of each "
    "participant — treat it as the sole source of truth. If the memory does "
    "not contain the answer, say you don't know rather than guessing."
)

QA_TEMPLATE = """\
Today's date is {question_date}.

Memory of the conversation:
{memory}

Using only the memory above, answer the question. Be concise and specific.

Question: {question}
Answer:"""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. You decide whether a model's answer to a "
    "question is correct given the gold answer."
)

JUDGE_TEMPLATE = """\
Question: {question}
Gold answer: {gold}
Model answer: {hypothesis}

Is the model answer correct? It counts as correct if it conveys the same key
information as the gold answer, even if phrased differently or with extra
detail. It is incorrect if it is missing the key information, contradicts the
gold answer, or declines to answer.

Respond with a single word: "yes" or "no"."""
