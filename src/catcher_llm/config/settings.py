from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env", override=True)

_LANGCHAIN_ENV_KEYS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
)
SUPPORTED_CHAT_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "ollama")
SUPPORTED_EMBEDDING_PROVIDERS: tuple[str, ...] = ("openai", "ollama")
_CHAT_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
}
_EMBEDDING_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "ollama": "ollama",
    "local": "ollama",
}


def _get_bool(name: str, default: bool) -> bool:
    """환경 변수 값을 불리언 설정값으로 해석한다."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(raw_value: str, aliases: dict[str, str], default: str) -> str:
    """provider 설정 문자열을 공백 제거, 소문자 변환, 별칭 치환 순서로 정규화한다."""
    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return default
    return aliases.get(normalized_value, normalized_value)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Catcher LLM"
    env_name: str = os.getenv("APP_ENV", "local")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
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
    def chat_provider(self) -> str:
        """채팅 모델 provider를 내부 표준 이름으로 반환한다."""
        return _normalize_provider(self.llm_provider, _CHAT_PROVIDER_ALIASES, "openai")

    @property
    def embedding_model_provider(self) -> str:
        """임베딩 모델 provider를 내부 표준 이름으로 반환한다."""
        return _normalize_provider(
            self.embedding_provider,
            _EMBEDDING_PROVIDER_ALIASES,
            "openai",
        )

    @property
    def chat_model_name(self) -> str:
        """현재 채팅 provider에 맞는 provider별 모델명을 반환한다."""
        provider = self.chat_provider
        if provider == "anthropic":
            return self.anthropic_model
        if provider == "ollama":
            return self.ollama_model
        return self.openai_model

    @property
    def embedding_model_name(self) -> str:
        """현재 임베딩 provider에 맞는 모델명을 반환한다."""
        if self.embedding_model_provider == "ollama":
            return self.ollama_embedding_model
        return self.embedding_model

    @property
    def chat_model_label(self) -> str:
        """UI와 로그에 표시할 채팅 모델 식별자를 반환한다."""
        return f"{self.chat_provider}:{self.chat_model_name}"

    @property
    def embedding_model_label(self) -> str:
        """UI와 로그에 표시할 임베딩 모델 식별자를 반환한다."""
        return f"{self.embedding_model_provider}:{self.embedding_model_name}"

    @property
    def has_openai_key(self) -> bool:
        """OpenAI API 키가 설정되어 있는지 확인한다."""
        return bool(self.openai_api_key.strip())

    @property
    def has_anthropic_key(self) -> bool:
        """Anthropic API 키가 설정되어 있는지 확인한다."""
        return bool(self.anthropic_api_key.strip())

    @property
    def chat_model_error(self) -> str | None:
        """채팅 모델 설정에 즉시 확인 가능한 오류가 있으면 오류 코드를 반환한다."""
        provider = self.chat_provider
        if provider not in SUPPORTED_CHAT_PROVIDERS:
            return "unsupported_llm_provider"
        if provider == "openai" and not self.has_openai_key:
            return "missing_openai_api_key"
        if provider == "anthropic" and not self.has_anthropic_key:
            return "missing_anthropic_api_key"
        return None

    @property
    def embedding_model_error(self) -> str | None:
        """임베딩 모델 설정에 즉시 확인 가능한 오류가 있으면 오류 코드를 반환한다."""
        provider = self.embedding_model_provider
        if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            return "unsupported_embedding_provider"
        if provider == "openai" and not self.has_openai_key:
            return "missing_openai_api_key"
        return None

    def get_chat_model_error_message(self, flow_name: str = "chat flow") -> str | None:
        """채팅 모델 설정 오류 코드를 사용자에게 보여줄 문장으로 변환한다."""
        error = self.chat_model_error
        if error == "missing_openai_api_key":
            return f"OPENAI_API_KEY is not set. Add it to .env before using the {flow_name}."
        if error == "missing_anthropic_api_key":
            return f"ANTHROPIC_API_KEY is not set. Add it to .env before using the {flow_name}."
        if error == "unsupported_llm_provider":
            supported = ", ".join(SUPPORTED_CHAT_PROVIDERS)
            return f"Unsupported LLM_PROVIDER '{self.llm_provider}'. Use one of: {supported}."
        return None

    def get_embedding_model_error_message(self, flow_name: str = "RAG flow") -> str | None:
        """임베딩 모델 설정 오류 코드를 사용자에게 보여줄 문장으로 변환한다."""
        error = self.embedding_model_error
        if error == "missing_openai_api_key":
            return f"OPENAI_API_KEY is not set. Add it to .env before using the {flow_name}."
        if error == "unsupported_embedding_provider":
            supported = ", ".join(SUPPORTED_EMBEDDING_PROVIDERS)
            return (
                f"Unsupported EMBEDDING_PROVIDER '{self.embedding_provider}'. "
                f"Use one of: {supported}."
            )
        return None

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
