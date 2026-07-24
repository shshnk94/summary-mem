from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.connection = self.prepare_db(self.db_path)

    def prepare_db(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path))
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

    def load(self, conversation_id: str, speaker_id: str) -> str | None:
        rows = self.connection.execute(
            "SELECT summary FROM summaries "
            "WHERE conversation_id = ? AND speaker_id = ?",
            (conversation_id, speaker_id),
        ).fetchall()
        return rows[0][0] if rows else None

    def load_all(self, conversation_id: str) -> dict[str, str]:

        query = f"SELECT speaker_id, summary FROM summaries WHERE conversation_id = '{conversation_id}' ORDER BY speaker_id"
        rows = self.connection.execute(query).fetchall()
        return {speaker: summary for speaker, summary in rows}

    def load_by_speaker(self, speaker_id: str) -> list[str]:
        """All of a speaker's summaries, one per conversation, ordered by conversation_id."""
        rows = self.connection.execute(
            "SELECT summary FROM summaries WHERE speaker_id = ? ORDER BY conversation_id",
            (speaker_id,),
        ).fetchall()
        return [summary for (summary,) in rows]

    def save(self, conversation_id: str, speaker_id: str, summary: str) -> None:
        self.connection.execute(
            "INSERT INTO summaries (conversation_id, speaker_id, summary) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(conversation_id, speaker_id) "
            "DO UPDATE SET summary = excluded.summary",
            (conversation_id, speaker_id, summary),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
