"""Shared defaults for summary-mem.

Mirrors the conventions used by the memLLM benchmark's memory mechanisms
(OpenRouter-hosted LLM + Qwen embeddings) so this package drops into the same
runner without per-mechanism config drift.
"""

from __future__ import annotations

# --- model / endpoint defaults (match memLLM) -------------------------------
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIMS = 4096
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

# --- retrieval / summarization knobs ----------------------------------------
# TODO: tune these once the mechanism is implemented.
DEFAULT_TOP_K = 5            # summaries returned by build_context
DEFAULT_SUMMARY_WINDOW = 20  # turns rolled into one leaf summary

# --- storage ------------------------------------------------------------
# MySQL database that rolling summaries are persisted to. Connection
# credentials are read from ~/.my.cnf (the same convention dlatk uses), so
# this only needs to name which database to use.
DEFAULT_DB_NAME = "ssubrahmanya"
