import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULTS = {
    "chat_model": "anthropic/claude-haiku-4.5",
    "adstrip_model": "anthropic/claude-haiku-4.5",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunk_tokens": 500,
    "chunk_overlap": 80,
    "languages": ["en"],
    "whisper_enabled_default": True,
    "whisper_model": "base",
    "request_delay": 1.0,
    "top_k": 6,
}


@dataclass
class Config:
    data_dir: Path
    openrouter_api_key: str | None
    chat_model: str
    adstrip_model: str
    embedding_model: str
    chunk_tokens: int
    chunk_overlap: int
    languages: list[str]
    whisper_enabled_default: bool
    whisper_model: str
    request_delay: float
    top_k: int


def default_data_dir() -> Path:
    return Path(os.environ.get("YTKB_DATA_DIR", "data")).resolve()


def load_config(data_dir: Path | None = None) -> Config:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    values = dict(DEFAULTS)
    toml_path = data_dir / "config.toml"
    if toml_path.exists():
        with toml_path.open("rb") as f:
            values.update(tomllib.load(f))

    env = dotenv_values(data_dir / ".env")
    api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    return Config(data_dir=data_dir, openrouter_api_key=api_key, **values)
