from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Provider: "openai" | "groq" | "gemini"
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    # Model names (provider-specific defaults)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    # Keys (set via environment variables)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # RAG / embeddings
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "local").strip().lower()
    openai_embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small").strip()
    docs_path: str = os.getenv("DOCS_PATH", "docs").strip()

    # Live web search (optional)
    # If provided, uses Serper (Google) API; otherwise falls back to DuckDuckGo (no key).
    serper_api_key: str | None = os.getenv("SERPER_API_KEY")


def get_config() -> AppConfig:
    return AppConfig()

