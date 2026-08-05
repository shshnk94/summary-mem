from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine

from .config import DEFAULT_DB_NAME

# Matches dlatk's convention: MySQL credentials live in ~/.my.cnf, so callers
# only ever need to name a database, not host/user/password.
MYSQL_CONFIG_FILE = Path.home() / ".my.cnf"


class Database:
    """Storage for rolling summaries, backed by MySQL via SQLAlchemy.

    A SQLAlchemy ``Engine`` pools connections per thread, so unlike a bare
    ``sqlite3.Connection`` this is safe to share across threads.
    """

    def __init__(
        self,
        db_name: str = DEFAULT_DB_NAME,
        table_name: str = "summaries",
        *,
        namespace: str,
    ) -> None:
        self.db_name = db_name
        self.table_name = table_name
        self.namespace = namespace
        self.engine = self._make_engine(db_name)
        self.metadata = MetaData()
        self.summaries = self._prepare_table()

    def _make_engine(self, db_name: str) -> Engine:
        connect_args = {}
        if MYSQL_CONFIG_FILE.is_file():
            connect_args["read_default_file"] = str(MYSQL_CONFIG_FILE)
        return create_engine(f"mysql+mysqldb:///{db_name}", connect_args=connect_args)

    def _prepare_table(self) -> Table:
        # Explicit InnoDB: the server default (MyISAM) caps composite-key
        # length at 1000 bytes, too short for three utf8mb4 VARCHAR(255) columns.
        # namespace scopes the key so unrelated pipelines/corpora sharing this
        # table (same db_name/table_name) can't collide on conversation_id/speaker_id.
        table = Table(
            self.table_name,
            self.metadata,
            Column("namespace", String(255), primary_key=True),
            Column("conversation_id", String(255), primary_key=True),
            Column("speaker_id", String(255), primary_key=True),
            Column("summary", Text, nullable=False),
            mysql_engine="InnoDB",
        )
        self.metadata.create_all(self.engine)
        return table

    def load(self, conversation_id: str, speaker_id: str) -> str | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(self.summaries.c.summary).where(
                    self.summaries.c.namespace == self.namespace,
                    self.summaries.c.conversation_id == conversation_id,
                    self.summaries.c.speaker_id == speaker_id,
                )
            ).first()
        return row[0] if row else None

    def load_all(self, conversation_id: str) -> dict[str, str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self.summaries.c.speaker_id, self.summaries.c.summary)
                .where(
                    self.summaries.c.namespace == self.namespace,
                    self.summaries.c.conversation_id == conversation_id,
                )
                .order_by(self.summaries.c.speaker_id)
            ).all()
        return {speaker: summary for speaker, summary in rows}

    def load_by_speaker(self, speaker_id: str) -> list[str]:
        """All of a speaker's summaries in this namespace, one per conversation, ordered by conversation_id."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self.summaries.c.summary)
                .where(
                    self.summaries.c.namespace == self.namespace,
                    self.summaries.c.speaker_id == speaker_id,
                )
                .order_by(self.summaries.c.conversation_id)
            ).all()
        return [summary for (summary,) in rows]

    def save(self, conversation_id: str, speaker_id: str, summary: str) -> None:
        stmt = mysql_insert(self.summaries).values(
            namespace=self.namespace,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            summary=summary,
        )
        stmt = stmt.on_duplicate_key_update(summary=stmt.inserted.summary)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def close(self) -> None:
        self.engine.dispose()
