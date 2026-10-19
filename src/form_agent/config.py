"""Settings from environment variables (and .env, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    reader: str = os.getenv("FORM_AGENT_READER", "openai")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
    data_dir: Path = Path(os.getenv("FORM_AGENT_DATA_DIR", "data"))
    cache_dir: Path = Path(os.getenv("FORM_AGENT_CACHE_DIR", "data/cache"))
    db_path: Path = Path(os.getenv("FORM_AGENT_DB", "data/forms.db"))


settings = Settings()
