"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    omdb_api_key: str
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tool_rounds: int = 5


def get_settings() -> Settings:
    """Create settings from the current environment."""

    api_key = os.getenv("OMDB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OMDB_API_KEY is missing. Create .env from .env.example and add your OMDb key."
        )

    return Settings(
        omdb_api_key=api_key,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "5")),
    )
