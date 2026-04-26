from __future__ import annotations

from pathlib import Path

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from catcher_llm.config.settings import Settings, get_settings
from catcher_llm.schemas.chat import ChatMessage, ChatRole

SESSION_MESSAGE_TABLE_NAME = "langchain_message_store"


def build_user_session_id(user_id: int, *, scope: str = "chat") -> str:
    """로그인 사용자와 기능 범위를 조합해 LangChain 세션 ID를 만든다."""
    normalized_scope = scope.strip().replace(" ", "_") or "chat"
    return f"user:{user_id}:{normalized_scope}"


def _build_sqlite_connection_string(sqlite_db_path: Path) -> str:
    """SQLChatMessageHistory가 사용할 SQLite 연결 문자열을 만든다."""
    return f"sqlite:///{sqlite_db_path.resolve().as_posix()}"


def get_sqlite_chat_message_history(
    session_id: str,
    *,
    settings: Settings | None = None,
) -> SQLChatMessageHistory:
    """설정된 세션 SQLite 파일에 연결된 LangChain 메시지 히스토리를 반환한다."""
    normalized_session_id = session_id.strip()
    if normalized_session_id == "":
        msg = "session_id must not be empty."
        raise ValueError(msg)

    config = settings or get_settings()
    sqlite_db_path = config.session_sqlite_db_path
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLChatMessageHistory(
        session_id=normalized_session_id,
        connection=_build_sqlite_connection_string(sqlite_db_path),
        table_name=SESSION_MESSAGE_TABLE_NAME,
    )


def _message_content_to_text(content: object) -> str:
    """LangChain 메시지 content 값을 Streamlit 표시용 문자열로 변환한다."""
    if isinstance(content, str):
        return content
    return str(content)


def _message_to_chat_role(message: BaseMessage) -> ChatRole | None:
    """LangChain 메시지 타입을 내부 채팅 역할 값으로 매핑한다."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return None


def load_session_chat_messages(
    session_id: str,
    *,
    settings: Settings | None = None,
) -> list[ChatMessage]:
    """SQLite에 저장된 LangChain 세션 메시지를 내부 채팅 메시지 목록으로 읽는다."""
    history = get_sqlite_chat_message_history(session_id, settings=settings)
    messages: list[ChatMessage] = []
    for message in history.messages:
        role = _message_to_chat_role(message)
        if role is None:
            continue
        messages.append(
            ChatMessage(
                role=role,
                content=_message_content_to_text(message.content),
            )
        )
    return messages


def append_session_chat_turn(
    session_id: str,
    *,
    user_input: str,
    ai_response: str,
    settings: Settings | None = None,
) -> None:
    """사용자 입력과 AI 응답 한 턴을 SQLite 세션 히스토리에 추가한다."""
    history = get_sqlite_chat_message_history(session_id, settings=settings)
    history.add_user_message(user_input)
    history.add_ai_message(ai_response)


def clear_session_chat_messages(
    session_id: str,
    *,
    settings: Settings | None = None,
) -> None:
    """지정한 LangChain 세션에 저장된 SQLite 메시지를 모두 삭제한다."""
    history = get_sqlite_chat_message_history(session_id, settings=settings)
    history.clear()
