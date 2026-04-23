from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    """환경 변수 값을 불리언 설정값으로 해석한다."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """환경 변수 값을 정수로 읽고, 변환에 실패하면 기본값을 반환한다."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Catcher LLM"
    env_name: str = os.getenv("APP_ENV", "local")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "catcher-llm")
    langsmith_dataset_name: str = os.getenv("LANGSMITH_DATASET", "catcher-llm-rag-eval")
    langsmith_experiment_prefix: str = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", "catcher-llm-rag")
    langsmith_tracing: bool = _get_bool("LANGSMITH_TRACING", True)
    rag_chunk_size: int = _get_int("RAG_CHUNK_SIZE", 800)
    rag_chunk_overlap: int = _get_int("RAG_CHUNK_OVERLAP", 120)
    rag_top_k: int = _get_int("RAG_TOP_K", 4)
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    vectorstore_dir: Path = DATA_DIR / "vectordb"
    eval_data_dir: Path = DATA_DIR / "evals"

    @property
    def has_openai_key(self) -> bool:
        """OpenAI API 키가 설정되어 있는지 확인한다."""
        return bool(self.openai_api_key.strip())

    @property
    def has_langsmith_key(self) -> bool:
        """LangSmith API 키가 설정되어 있는지 확인한다."""
        return bool(self.langsmith_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """환경 변수 기반 앱 설정을 한 번 생성한 뒤 캐시해서 반환한다."""
    return Settings()
