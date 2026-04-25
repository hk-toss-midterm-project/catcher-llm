from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.rag_prompt import build_rag_prompt


def build_rag_chain(settings: Settings | None = None, *, temperature: float = 0.0):
    """RAG 프롬프트, 채팅 모델, 문자열 파서를 연결한 답변 생성 체인을 만든다."""
    return (
        build_rag_prompt() | get_chat_model(settings, temperature=temperature) | StrOutputParser()
    )
