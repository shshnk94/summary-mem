"""Prompts for the running per-speaker summary.

See ``memory.py``: each speaker has a summary that is rewritten from
(existing summary + new turns) every time that speaker produces turns.
"""

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a concise, factual running summary of a single speaker in a "
    "conversation. The summary captures that speaker's stable state: their "
    "facts, situation, relationships, preferences, decisions, and goals, plus "
    "anything they have revealed about themselves. Integrate the new turns into "
    "the existing summary, update or correct anything that has changed, and keep "
    "everything that still holds. Write in the third person and return only the "
    "updated summary text, with no preamble."
)

UPDATE_TEMPLATE = """\
Existing summary of {speaker}:
{existing}

New turns spoken by {speaker}:
{turns}

Return the updated summary of {speaker}."""
