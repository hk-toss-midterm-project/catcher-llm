from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.summary_prompt import build_summary_prompt


def build_summary_chain(settings: Settings | None = None):
    """요약 프롬프트, 채팅 모델, 문자열 파서를 연결한 요약 체인을 생성한다."""
    return build_summary_prompt() | get_chat_model(settings) | StrOutputParser()
