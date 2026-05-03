from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from pydantic import SecretStr

from catcher_llm.config.settings import Settings, get_settings


def get_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
) -> BaseChatModel:
    """설정 provider와 호출 옵션 temperature로 LangChain 채팅 모델을 생성한다."""
    config = settings or get_settings()
    config_error = config.chat_model_error
    if config_error is not None:
        error_message = config.get_chat_model_error_message() or config_error
        raise ValueError(error_message)

    provider = config.chat_provider
    if provider == "openai":
        return ChatOpenAI(
            api_key=SecretStr(config.openai_api_key),
            model=config.chat_model_name,
            temperature=temperature,
        )
    if provider == "anthropic":
        return ChatAnthropic(
            api_key=SecretStr(config.anthropic_api_key),
            model_name=config.chat_model_name,
            temperature=temperature,
            timeout=None,
            stop=None,
        )
    if provider == "ollama":
        return ChatOllama(
            model=config.chat_model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
        )

    supported = ", ".join(("openai", "anthropic", "ollama"))
    raise ValueError(f"Unsupported LLM_PROVIDER '{config.llm_provider}'. Use one of: {supported}.")


def get_embeddings_model(settings: Settings | None = None) -> Embeddings:
    """설정에 지정된 provider와 모델명으로 LangChain 임베딩 모델 인스턴스를 생성한다."""
    config = settings or get_settings()
    config_error = config.embedding_model_error
    if config_error is not None:
        error_message = config.get_embedding_model_error_message() or config_error
        raise ValueError(error_message)

    provider = config.embedding_model_provider
    if provider == "openai":
        return OpenAIEmbeddings(
            api_key=SecretStr(config.openai_api_key),
            model=config.embedding_model_name,
        )
    if provider == "ollama":
        return OllamaEmbeddings(
            model=config.embedding_model_name,
            base_url=config.ollama_base_url,
        )
    if provider == "upstage":
        return UpstageEmbeddings(
            upstage_api_key=SecretStr(config.upstage_api_key),
            model=config.embedding_model_name,
        )

    supported = ", ".join(("openai", "ollama", "upstage"))
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{config.embedding_provider}'. Use one of: {supported}."
    )
