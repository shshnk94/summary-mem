"""Vector store over summary nodes.

Self-contained in-memory store (numpy cosine search), optionally persisted to
`output_dir`/`conversation_id` for disk-backed runs. Holds the embeddings that
back dense retrieval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    """Append-only embedding store keyed by node id."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self.persist_dir = persist_dir
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add(self, ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        """Append nodes and their (pre-normalized) embeddings."""
        # TODO: stack embeddings; extend id/text lists; persist if persist_dir set.
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return [(node_id, score)] for the top_k nearest nodes by cosine."""
        raise NotImplementedError

    def clear(self) -> None:
        """Drop all stored vectors, keeping the store object reusable."""
        self.__init__(self.persist_dir)
