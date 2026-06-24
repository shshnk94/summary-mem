"""Shared defaults for summary-mem.

Mirrors the conventions used by the memLLM benchmark's memory mechanisms
(OpenRouter-hosted LLM + Qwen embeddings) so this package drops into the same
runner without per-mechanism config drift.
"""

from __future__ import annotations

# --- model / endpoint defaults (match memLLM) -------------------------------
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIMS = 4096
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- retrieval / summarization knobs ----------------------------------------
# TODO: tune these once the mechanism is implemented.
DEFAULT_TOP_K = 5            # summaries returned by build_context
DEFAULT_SUMMARY_WINDOW = 20  # turns rolled into one leaf summary
