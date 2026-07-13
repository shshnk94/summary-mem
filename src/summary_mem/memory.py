from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from .config import DEFAULT_LLM_MODEL
from .database import Database
from .prompts.summarization import SUMMARY_PROMPT


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

    def close(self) -> None:
        self.db.close()

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
        return json.loads(response.choices[0].message.content)["summary"].strip()