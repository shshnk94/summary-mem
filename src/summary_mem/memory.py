from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry, 
    retry_if_exception_type, 
    stop_after_attempt, 
    wait_random_exponential
)

from .database import BatchDatabase, Database
from .prompts.summarization import BATCH_SUMMARY, SUMMARY

# --summary_prompt name -> (prompt template, table it's stored under).
SUMMARY_PROMPTS = {
    "plain": (SUMMARY, "summaries_plain"),
}


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    print(
        f"summarize() API error on attempt {retry_state.attempt_number}: {exc!r}",
        file=sys.stderr,
    )


class BaseMemory(ABC):
    """Common interface implemented by every memory mechanism.

    ``update`` folds a new turn into memory; ``recall_speaker`` reads it back at
    the speaker granularity; ``close`` releases any underlying resources
    (e.g. database connections).
    """

    @abstractmethod
    def update(
        self,
        conversation_id: str,
        speaker_id: str,
        turn_text: str
    ) -> None:
        pass

    @abstractmethod
    def recall_speaker(self, speaker_id: str) -> list[str]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SummaryMemory(BaseMemory):

    def __init__(
        self,
        chat_client: OpenAI,
        namespace: str,
        db_name: str = "ssubrahmanya",
        model: str = "openai/gpt-4o-mini",
        summary_prompt: str = "plain",
        max_sum_tokens: int = 500,
        temperature: float = 0.0,
    ) -> None:

        self.chat_client = chat_client
        self.model = model
        self.max_sum_tokens = max_sum_tokens
        self.temperature = temperature
        self.prompt_template, table_name = SUMMARY_PROMPTS[summary_prompt]
        self.db = Database(
            db_name,
            table_name=table_name,
            namespace=namespace
        )

    def update(self, conversation_id: str, speaker_id: str, turn_text: str) -> None:

        # load the database to memory
        existing = self.db.load(conversation_id, speaker_id)

        # update the summary with the new turn
        updated = self.summarize(speaker_id, existing, turn_text)

        # save the updated summary back to the database
        self.db.save(conversation_id, speaker_id, updated)

    def recall_speaker(self, speaker_id: str) -> list[str]:
        """A speaker's rolling summaries across every conversation they appear in."""
        return self.db.load_by_speaker(speaker_id)

    def close(self) -> None:
        self.db.close()

    @retry(
        retry=retry_if_exception_type(
            (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                json.JSONDecodeError,
                KeyError,
            )
        ),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def summarize(
        self,
        speaker: str,
        existing: str | None,
        turn_text: str
    ) -> str:

        prompt = self.prompt_template.format(
            speaker=speaker,
            existing=existing or "(no summary yet)",
            turns=turn_text,
            max_sum_words=self.max_sum_tokens,
        )
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)["summary"].strip()
        except (json.JSONDecodeError, KeyError):
            print(f"summarize() malformed response ({len(content)} chars): {content[:500]!r}", file=sys.stderr)
            raise


class InContextMemory(BaseMemory):
    """No-summarization baseline: keeps every turn verbatim instead of rolling
    them into a summary. Same interface as ``SummaryMemory`` (update/recall_speaker/
    close), so callers can use either mechanism interchangeably.
    """

    def __init__(self) -> None:
        self.memory: dict[str, dict[str, list[str]]] = {}

    def update(
        self, 
        conversation_id: str, 
        speaker_id: str, 
        turn_text: str
    ) -> None:
        
        if speaker_id not in self.memory:
            self.memory[speaker_id] = {}
        if conversation_id not in self.memory[speaker_id]:
            self.memory[speaker_id][conversation_id] = []

        self.memory[speaker_id][conversation_id].append(turn_text)

    def recall_speaker(self, speaker_id: str) -> list[str]:

        conversations = self.memory.get(speaker_id, {})
        turns = []
        for conversation_id in sorted(conversations):
            cturns = "\n".join(conversations[conversation_id])
            turns.append(cturns)

        return turns

    def close(self) -> None:
        pass


class BatchSummaryMemory(BaseMemory):
    """One-shot baseline: summarizes all of a speaker's turns, pooled across
    every conversation they appear in, with a single call -- instead of
    folding turns into a running summary one at a time like ``SummaryMemory``.
    Same interface (update/recall_speaker/close), so callers can swap it in
    for either mechanism.
    """

    def __init__(
        self,
        chat_client: OpenAI,
        namespace: str,
        db_name: str = "ssubrahmanya",
        model: str = "openai/gpt-4o-mini",
        max_sum_tokens: int = 500,
        temperature: float = 0.0,
    ) -> None:

        self.chat_client = chat_client
        self.model = model
        self.max_sum_tokens = max_sum_tokens
        self.temperature = temperature
        self.prompt_template = BATCH_SUMMARY
        self.db = BatchDatabase(db_name, table_name="summary_batch", namespace=namespace)
        self.turns: dict[str, dict[str, list[str]]] = {}
        self.summaries: dict[str, str] = {}

    def update(
        self,
        conversation_id: str,
        speaker_id: str,
        turn_text: str,
    ) -> None:

        if speaker_id not in self.turns:
            self.turns[speaker_id] = {}
        if conversation_id not in self.turns[speaker_id]:
            self.turns[speaker_id][conversation_id] = []

        self.turns[speaker_id][conversation_id].append(turn_text)

    def recall_speaker(self, speaker_id: str) -> list[str]:
        """A speaker's single summary, pooled across every conversation they appear in."""

        if speaker_id not in self.summaries:
            conversations = self.turns.get(speaker_id, {})
            turns = []
            for conversation_id in sorted(conversations):
                turns.extend(conversations[conversation_id])

            summary = self.summarize(speaker_id, "\n".join(turns))
            self.db.save(speaker_id, summary)
            self.summaries[speaker_id] = summary

        context = [self.summaries[speaker_id]]
        return context

    def close(self) -> None:
        self.db.close()

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            json.JSONDecodeError,
            KeyError,
        )),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def summarize(self, speaker: str, turns: str) -> str:
        prompt = self.prompt_template.format(speaker=speaker, turns=turns, max_sum_words=self.max_sum_tokens)
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)["summary"].strip()
        except (json.JSONDecodeError, KeyError):
            print(f"summarize() malformed response ({len(content)} chars): {content[:500]!r}", file=sys.stderr)
            raise


class ConversationBatchSummaryMemory(BaseMemory):
    """One-shot baseline like ``BatchSummaryMemory``, but scoped per conversation
    instead of pooled across a speaker's entire history: each (conversation_id,
    speaker_id) pair gets its own single-pass summary of just that conversation's
    turns, with no recursive merging. Same interface (update/recall_speaker/
    close), so callers can swap it in for either mechanism. ``recall_speaker`` then
    pools those per-conversation summaries across every conversation the speaker
    appears in, ordered by conversation_id -- the same shape ``SummaryMemory``
    returns, just with one-shot rather than recursively-merged summaries.
    """

    def __init__(
        self,
        chat_client: OpenAI,
        namespace: str,
        db_name: str = "ssubrahmanya",
        model: str = "openai/gpt-4o-mini",
        max_sum_tokens: int = 500,
        temperature: float = 0.0,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.max_sum_tokens = max_sum_tokens
        self.temperature = temperature
        self.prompt_template = BATCH_SUMMARY
        self.db = Database(db_name, table_name="summary_conversation_batch", namespace=namespace)
        self.turns: dict[tuple[str, str], list[str]] = {}
        self.summaries: dict[tuple[str, str], str] = {}

    def update(
        self,
        conversation_id: str,
        speaker_id: str,
        turn_text: str,
    ) -> None:

        key = (conversation_id, speaker_id)
        if key not in self.turns:
            self.turns[key] = []

        self.turns[key].append(turn_text)

    def recall_speaker(self, speaker_id: str) -> list[str]:
        """A speaker's one-shot summaries, one per conversation they appear in, ordered by conversation_id."""

        for conversation_id, sid in list(self.turns):
            if sid != speaker_id:
                continue

            key = (conversation_id, sid)
            if key not in self.summaries:
                turns = "\n".join(self.turns.get(key, []))
                summary = self.summarize(sid, turns)
                self.db.save(conversation_id, sid, summary)
                self.summaries[key] = summary

        return self.db.load_by_speaker(speaker_id)

    def close(self) -> None:
        self.db.close()

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            json.JSONDecodeError,
            KeyError,
        )),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def summarize(self, speaker: str, turns: str) -> str:
        prompt = self.prompt_template.format(speaker=speaker, turns=turns, max_sum_words=self.max_sum_tokens)
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)["summary"].strip()
        except (json.JSONDecodeError, KeyError):
            print(f"summarize() malformed response ({len(content)} chars): {content[:500]!r}", file=sys.stderr)
            raise
