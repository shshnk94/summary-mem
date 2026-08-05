from __future__ import annotations

import json
import sys

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import DEFAULT_DB_NAME, DEFAULT_LLM_MODEL
from .database import Database
from .prompts.summarization import QUESTIONNAIRE_SUMMARY


class MalformedSummaryResponse(Exception):
    """The model's response wasn't the expected {"summary": ...} JSON object.

    Carries the raw response text so it can be inspected — this happens when the
    model emits a runaway response (e.g. thousands of repeated lines) that gets cut
    off mid-JSON, rather than a well-formed summary.
    """

    def __init__(self, content: str, cause: Exception) -> None:
        super().__init__(f"{cause} ({len(content)} chars)")
        self.content = content


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    if isinstance(exc, MalformedSummaryResponse):
        print(
            f"summarize() malformed response on attempt {retry_state.attempt_number} "
            f"({exc}):\n--- response content (first 2000 chars) ---\n{exc.content[:2000]}\n"
            "--- end response content ---",
            file=sys.stderr,
        )
    else:
        print(
            f"summarize() API error on attempt {retry_state.attempt_number}: {exc!r}",
            file=sys.stderr,
        )


class SummaryMemory:

    def __init__(
        self,
        chat_client: OpenAI,
        db_name: str = DEFAULT_DB_NAME,
        model: str = DEFAULT_LLM_MODEL,
        prompt_template: str = QUESTIONNAIRE_SUMMARY,
        table_name: str = "summaries",
        *,
        namespace: str,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.prompt_template = prompt_template
        self.db = Database(db_name, table_name=table_name, namespace=namespace)

    def update(self, conversation_id: str, speaker_id: str, turn_text: str) -> None:

        # load the database to memory
        existing = self.db.load(conversation_id, speaker_id)

        # update the summary with the new turn
        updated = self.summarize(speaker_id, existing, turn_text)

        # save the updated summary back to the database
        self.db.save(conversation_id, speaker_id, updated)

    def recall(self, conversation_id: str) -> dict[str, str]:
        return self.db.load_all(conversation_id)

    def recall_all(self, speaker_id: str) -> list[str]:
        """A speaker's rolling summaries across every conversation they appear in."""
        return self.db.load_by_speaker(speaker_id)

    def close(self) -> None:
        self.db.close()

    @retry(
        retry=retry_if_exception_type((
            MalformedSummaryResponse,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        )),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        reraise=True,
    )
    def summarize(self, speaker: str, existing: str | None, turn_text: str) -> str:
        prompt = self.prompt_template.format(
            speaker=speaker,
            existing=existing or "(no summary yet)",
            turns=turn_text,
        )
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)["summary"].strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MalformedSummaryResponse(content, exc) from exc


class InContextMemory:
    """No-summarization baseline: keeps every turn verbatim instead of rolling
    them into a summary. Same interface as ``SummaryMemory`` (update/recall/
    recall_all/close), so callers can use either mechanism interchangeably.
    """

    def __init__(self) -> None:
        self._turns: dict[tuple[str, str], list[str]] = {}

    def update(self, conversation_id: str, speaker_id: str, turn_text: str) -> None:
        self._turns.setdefault((conversation_id, speaker_id), []).append(turn_text)

    def recall(self, conversation_id: str) -> dict[str, str]:
        return {
            speaker_id: "\n".join(turns)
            for (cid, speaker_id), turns in self._turns.items()
            if cid == conversation_id
        }

    def recall_all(self, speaker_id: str) -> list[str]:
        """A speaker's turns across every conversation they appear in, one blob per conversation."""
        return [
            "\n".join(turns)
            for (conversation_id, sid), turns in sorted(self._turns.items())
            if sid == speaker_id
        ]

    def close(self) -> None:
        pass
