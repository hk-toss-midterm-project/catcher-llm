from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

_LANGCHAIN_ENV_KEYS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
)


def _get_bool(name: str, default: bool) -> bool:
    """환경 변수 값을 불리언 설정값으로 해석한다."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Catcher LLM"
    env_name: str = os.getenv("APP_ENV", "local")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "catcher-llm")
    langsmith_dataset_name: str = os.getenv("LANGSMITH_DATASET", "catcher-llm-rag-eval")
    langsmith_experiment_prefix: str = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", "catcher-llm-rag")
    langsmith_tracing: bool = _get_bool("LANGSMITH_TRACING", True)
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    vectorstore_dir: Path = DATA_DIR / "vectordb"
    eval_data_dir: Path = DATA_DIR / "evals"
    sqlite_db_path: Path = DATA_DIR / "sqlite" / "app.sqlite3"

    @property
    def has_openai_key(self) -> bool:
        """OpenAI API 키가 설정되어 있는지 확인한다."""
        return bool(self.openai_api_key.strip())

    @property
    def has_langsmith_key(self) -> bool:
        """LangSmith API 키가 설정되어 있는지 확인한다."""
        return bool(self.langsmith_api_key.strip())

    @property
    def members_csv_path(self) -> Path:
        """로컬 사용자 시드 CSV 경로를 반환한다."""
        return self.raw_data_dir / "csv" / "members_v1.csv"

    @property
    def consumption_csv_path(self) -> Path:
        """로컬 소비내역 시드 CSV 경로를 반환한다."""
        return self.raw_data_dir / "csv" / "consumption_v1.csv"

    @property
    def session_sqlite_db_path(self) -> Path:
        """랭체인 세션/대화 영속화를 위한 SQLite 경로를 반환한다."""
        return self.sqlite_db_path.with_name("session.sqlite3")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """환경 변수 기반 앱 설정을 한 번 생성한 뒤 캐시해서 반환한다."""
    return Settings()


def configure_langsmith_env(settings: Settings | None = None) -> Settings:
    """LangSmith tracing에 필요한 LangChain 환경변수를 현재 설정값으로 반영한다."""
    config = settings or get_settings()
    if config.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith_endpoint
    else:
        for key in _LANGCHAIN_ENV_KEYS:
            os.environ.pop(key, None)

    return config
