from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.consumption_feedback.memory import (
    build_feedback_memory_rank_prompt,
    build_memory_summary_prompt,
)


def build_memory_summary_chain(
    period_label: str,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], str]:
    """기간별 피드백 세션 목록을 장기 메모리 요약 문자열로 변환하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_memory_summary_prompt(period_label) | chat_model | StrOutputParser()


def build_feedback_memory_rank_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], str]:
    """[거부/제약] 항목 목록을 의미 기반 구체성 순으로 재정렬하는 체인을 만든다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_feedback_memory_rank_prompt() | chat_model | StrOutputParser()


__all__ = [
    "build_memory_summary_chain",
    "build_feedback_memory_rank_chain",
]
