from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Matches dlatk's convention: MySQL credentials live in ~/.my.cnf, so callers
# only ever need to name a database, not host/user/password.
MYSQL_CONFIG_FILE = Path.home() / ".my.cnf"


def _make_engine(db_name: str) -> Engine:
    connect_args = {}
    if MYSQL_CONFIG_FILE.is_file():
        connect_args["read_default_file"] = str(MYSQL_CONFIG_FILE)
    return create_engine(f"mysql+mysqldb:///{db_name}", connect_args=connect_args)


class Database:
    """Storage for rolling summaries, backed by MySQL via SQLAlchemy.

    A SQLAlchemy ``Engine`` pools connections per thread, so unlike a bare
    ``sqlite3.Connection`` this is safe to share across threads.
    """

    def __init__(
        self,
        db_name: str = "ssubrahmanya",
        table_name: str = "summaries",
        *,
        namespace: str,
    ) -> None:
        self.db_name = db_name
        self.table_name = table_name
        self.namespace = namespace
        self.engine = _make_engine(db_name)
        self._create_table()

    def _create_table(self) -> None:
        # Explicit InnoDB: the server default (MyISAM) caps composite-key
        # length at 1000 bytes, too short for three utf8mb4 VARCHAR(255) columns.
        # namespace scopes the key so unrelated pipelines/corpora sharing this
        # table (same db_name/table_name) can't collide on conversation_id/speaker_id.
        query = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"namespace VARCHAR(255) NOT NULL, "
            f"conversation_id VARCHAR(255) NOT NULL, "
            f"speaker_id VARCHAR(255) NOT NULL, "
            f"summary TEXT NOT NULL, "
            f"PRIMARY KEY (namespace, conversation_id, speaker_id)"
            f") ENGINE=InnoDB"
        )
        with self.engine.begin() as conn:
            conn.execute(text(query))

    def load(self, conversation_id: str, speaker_id: str) -> str | None:
        query = (
            f"SELECT summary FROM {self.table_name} "
            f"WHERE namespace = '{self.namespace}' "
            f"AND conversation_id = '{conversation_id}' "
            f"AND speaker_id = '{speaker_id}'"
        )
        with self.engine.connect() as conn:
            row = conn.execute(text(query)).first()
        return row[0] if row else None

    def load_all(self, conversation_id: str) -> dict[str, str]:
        query = (
            f"SELECT speaker_id, summary FROM {self.table_name} "
            f"WHERE namespace = '{self.namespace}' "
            f"AND conversation_id = '{conversation_id}' "
            f"ORDER BY speaker_id"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(query)).all()
        return {speaker: summary for speaker, summary in rows}

    def load_by_speaker(self, speaker_id: str) -> list[str]:
        """All of a speaker's summaries in this namespace, one per conversation, ordered by conversation_id."""
        query = (
            f"SELECT summary FROM {self.table_name} "
            f"WHERE namespace = '{self.namespace}' "
            f"AND speaker_id = '{speaker_id}' "
            f"ORDER BY conversation_id"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(query)).all()
        return [summary for (summary,) in rows]

    def save(self, conversation_id: str, speaker_id: str, summary: str) -> None:
        query = (
            f"INSERT INTO {self.table_name} (namespace, conversation_id, speaker_id, summary) "
            f"VALUES ('{self.namespace}', '{conversation_id}', '{speaker_id}', :summary) "
            f"ON DUPLICATE KEY UPDATE summary = VALUES(summary)"
        )
        with self.engine.begin() as conn:
            conn.execute(text(query), {"summary": summary})

    def delete_conversations(self, conversation_ids: list[str]) -> None:
        """Wipe this namespace's rows for the given conversation_ids (e.g. before a re-run)."""
        ids = "', '".join(conversation_ids)
        query = (
            f"DELETE FROM {self.table_name} "
            f"WHERE namespace = '{self.namespace}' "
            f"AND conversation_id IN ('{ids}')"
        )
        with self.engine.begin() as conn:
            conn.execute(text(query))

    def close(self) -> None:
        self.engine.dispose()


class BatchDatabase:
    """Storage for one-shot, per-speaker summaries (see ``BatchSummaryMemory``).

    Unlike ``Database``, rows are pooled across every conversation a speaker
    appears in, so they're keyed by speaker alone -- there's no conversation_id.
    """

    def __init__(
        self,
        db_name: str = "ssubrahmanya",
        table_name: str = "summary_batch",
        *,
        namespace: str,
    ) -> None:
        self.db_name = db_name
        self.table_name = table_name
        self.namespace = namespace
        self.engine = _make_engine(db_name)
        self._create_table()

    def _create_table(self) -> None:
        query = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"namespace VARCHAR(255) NOT NULL, "
            f"speaker_id VARCHAR(255) NOT NULL, "
            f"summary TEXT NOT NULL, "
            f"PRIMARY KEY (namespace, speaker_id)"
            f") ENGINE=InnoDB"
        )
        with self.engine.begin() as conn:
            conn.execute(text(query))

    def load(self, speaker_id: str) -> str | None:
        query = (
            f"SELECT summary FROM {self.table_name} "
            f"WHERE namespace = '{self.namespace}' "
            f"AND speaker_id = '{speaker_id}'"
        )
        with self.engine.connect() as conn:
            row = conn.execute(text(query)).first()
        return row[0] if row else None

    def save(self, speaker_id: str, summary: str) -> None:
        query = (
            f"INSERT INTO {self.table_name} (namespace, speaker_id, summary) "
            f"VALUES ('{self.namespace}', '{speaker_id}', :summary) "
            f"ON DUPLICATE KEY UPDATE summary = VALUES(summary)"
        )
        with self.engine.begin() as conn:
            conn.execute(text(query), {"summary": summary})

    def close(self) -> None:
        self.engine.dispose()
