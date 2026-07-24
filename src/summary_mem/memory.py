from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import DEFAULT_LLM_MODEL
from .database import Database
from .prompts.summarization import SUMMARY_PROMPT


class MalformedSummaryResponse(Exception):
    """The model's response wasn't the expected {"summary": ...} JSON object.

    Carries the raw response text so it can be inspected — this happens when the
    model emits a runaway response (e.g. thousands of repeated lines) that gets cut
    off mid-JSON, rather than a well-formed summary.
    """

    def __init__(self, content: str, cause: Exception) -> None:
        super().__init__(f"{cause} ({len(content)} chars)")
        self.content = content


def _log_malformed_response(retry_state) -> None:
    exc = retry_state.outcome.exception()
    print(
        f"summarize() malformed response on attempt {retry_state.attempt_number} "
        f"({exc}):\n--- response content (first 2000 chars) ---\n{exc.content[:2000]}\n"
        "--- end response content ---",
        file=sys.stderr,
    )


class SummaryMemory:

    def __init__(
        self,
        chat_client: OpenAI,
        db_path: Path,
        model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.db = Database(db_path)

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
        retry=retry_if_exception_type(MalformedSummaryResponse),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(3),
        before_sleep=_log_malformed_response,
        reraise=True,
    )
    def summarize(self, speaker: str, existing: str | None, turn_text: str) -> str:
        prompt = SUMMARY_PROMPT.format(
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
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)["summary"].strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MalformedSummaryResponse(content, exc) from exc