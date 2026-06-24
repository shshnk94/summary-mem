from __future__ import annotations

import sqlite3
from pathlib import Path

from openai import OpenAI

from .config import DEFAULT_LLM_MODEL
from .prompts.summarization import SUMMARY_SYSTEM_PROMPT, UPDATE_TEMPLATE


class SummaryMemory:
    """Running per-speaker conversation summary backed by an LLM chat client."""

    def __init__(
        self,
        chat_client: OpenAI,
        *,
        model: str = DEFAULT_LLM_MODEL,
        db_path: str | Path | None = None,
    ) -> None:
        """Wire up the LLM client and the summary database.

        Args:
            chat_client: An OpenAI-compatible chat client (see ``clients.py``).
            model: Chat model used to (re)write summaries.
            db_path: SQLite file backing the summaries. Defaults to an
                in-memory database that lives for the object's lifetime.
        """
        self.chat_client = chat_client
        self.model = model
        self._db_path = str(db_path) if db_path is not None else ":memory:"
        self._db = self.prepare_db()

    def prepare_db(self) -> sqlite3.Connection:
        """Open the SQLite database backing this memory and return the connection.

        One storage holds every conversation this memory has seen, keyed by
        ``(conversation_id, speaker_id)``. If the database file already exists,
        it is reused as-is; otherwise its parent directories and the
        ``summaries`` table are created.
        """
        if self._db_path != ":memory:" and Path(self._db_path).exists():
            return sqlite3.connect(self._db_path)

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._db_path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS summaries ("
            "  conversation_id TEXT NOT NULL,"
            "  speaker_id      TEXT NOT NULL,"
            "  summary         TEXT NOT NULL,"
            "  PRIMARY KEY (conversation_id, speaker_id)"
            ")"
        )
        connection.commit()
        return connection

    def update(self, conversation_id: str, speaker_id: str, turn_text: str) -> None:
        """Fold a single new turn into the speaker's running summary.

        Loads the current ``(conversation_id, speaker_id)`` summary, rewrites it
        with the information from ``turn_text``, and stores the result back.
        Only the producing speaker's state is touched.
        """
        existing = self._load(conversation_id, speaker_id)
        updated = self._summarize(speaker_id, existing, turn_text)
        self._save(conversation_id, speaker_id, updated)

    def recall(self, conversation_id: str) -> dict[str, str]:
        """Return the current summary for every speaker in the conversation.

        The returned mapping ``{speaker_id: summary}`` covers both sides of the
        conversation so the generative LLM has the entire state to condition on.
        """
        rows = self._db.execute(
            "SELECT speaker_id, summary FROM summaries "
            "WHERE conversation_id = ? ORDER BY speaker_id",
            (conversation_id,),
        ).fetchall()
        return {speaker: summary for speaker, summary in rows}

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

    # -- internals -----------------------------------------------------------

    def _summarize(self, speaker: str, existing: str | None, turn_text: str) -> str:
        """Call the LLM to fold a new turn into the speaker's running summary."""
        prompt = UPDATE_TEMPLATE.format(
            speaker=speaker,
            existing=existing or "(no summary yet)",
            turns=turn_text,
        )
        response = self.chat_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _load(self, conversation_id: str, speaker: str) -> str | None:
        """Fetch a speaker's stored summary, or None if none exists yet."""
        row = self._db.execute(
            "SELECT summary FROM summaries "
            "WHERE conversation_id = ? AND speaker_id = ?",
            (conversation_id, speaker),
        ).fetchone()
        return row[0] if row else None

    def _save(self, conversation_id: str, speaker: str, summary: str) -> None:
        """Upsert a speaker's summary under the composite key."""
        self._db.execute(
            "INSERT INTO summaries (conversation_id, speaker_id, summary) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(conversation_id, speaker_id) "
            "DO UPDATE SET summary = excluded.summary",
            (conversation_id, speaker, summary),
        )
        self._db.commit()