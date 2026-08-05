"""Thin client helpers for the LLM and embedding endpoints.

Self-contained: depends only on `openai` (OpenRouter-compatible), not on the
hipporag or mem0 packages. API keys are read from a `.env` file via
`python-dotenv`, exactly as the memLLM mechanisms do.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from .config import OPENROUTER_BASE_URL, VLLM_BASE_URL

load_dotenv()

def get_chat_client(base_url: str = VLLM_BASE_URL) -> OpenAI:
    """OpenAI client for chat/summarization calls (local vLLM server by default)."""
    return OpenAI(
        base_url=base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def get_embedding_client(base_url: str = OPENROUTER_BASE_URL) -> OpenAI:
    """OpenAI client for embedding calls (OpenRouter by default)."""
    return OpenAI(
        base_url=base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )