from __future__ import annotations

from collections.abc import Callable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from catcher_llm.config.settings import Settings
from catcher_llm.llm.models import get_chat_model
from catcher_llm.prompts.chat_prompt import build_chat_prompt, build_session_chat_prompt


def build_chat_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
):
    """채팅 프롬프트, LLM, 문자열 파서를 연결한 기본 대화 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    return build_chat_prompt() | chat_model | StrOutputParser()


def _build_session_history_factory(
    settings: Settings | None,
) -> Callable[[str], BaseChatMessageHistory]:
    """세션 ID를 받아 설정된 SQLite 메시지 히스토리를 여는 팩토리를 만든다."""

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """RunnableWithMessageHistory가 호출할 세션별 히스토리를 반환한다."""
        from catcher_llm.services.session_history_service import get_sqlite_chat_message_history

        return get_sqlite_chat_message_history(session_id, settings=settings)

    return get_session_history


def build_session_chat_chain(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    *,
    temperature: float = 0.0,
) -> Runnable[dict[str, str], str]:
    """LangChain 메시지 히스토리를 SQLite에 저장하는 세션 채팅 체인을 생성한다."""
    chat_model = llm or get_chat_model(settings, temperature=temperature)
    chain = build_session_chat_prompt() | chat_model | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        _build_session_history_factory(settings),
        input_messages_key="input",
        history_messages_key="history",
    )
