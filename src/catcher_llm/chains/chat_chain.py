from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.chat_prompt import build_chat_prompt


def build_chat_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
):
    """채팅 프롬프트, LLM, 문자열 파서를 연결한 기본 대화 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_chat_prompt() | chat_model | StrOutputParser()
