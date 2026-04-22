from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from catcher_llm.config.settings import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    config = settings or get_settings()
    return ChatOpenAI(
        api_key=SecretStr(config.openai_api_key),
        model=config.openai_model,
        temperature=0,
    )


def get_embeddings_model(settings: Settings | None = None) -> OpenAIEmbeddings:
    config = settings or get_settings()
    return OpenAIEmbeddings(
        api_key=SecretStr(config.openai_api_key),
        model=config.embedding_model,
    )
