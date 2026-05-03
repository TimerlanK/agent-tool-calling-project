"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    omdb_api_key: str
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"
    temperature: float = 0.2
    max_tool_rounds: int = 8
    memory_max_word_limit: int = 120
    memory_keep_last_messages: int = 4


def get_settings() -> Settings:
    """Create settings from the current environment."""

    omdb_api_key = os.getenv("OMDB_API_KEY", "").strip()
    if not omdb_api_key:
        raise RuntimeError(
            "OMDB_API_KEY is missing. Create .env from .env.example and add your OMDb key."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create .env from .env.example and add your OpenAI API key."
        )

    return Settings(
        omdb_api_key=omdb_api_key,
        openai_api_key=openai_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "8")),
        memory_max_word_limit=int(os.getenv("MEMORY_MAX_WORD_LIMIT", "120")),
        memory_keep_last_messages=int(os.getenv("MEMORY_KEEP_LAST_MESSAGES", "4")),
    )
