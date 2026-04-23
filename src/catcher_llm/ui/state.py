from __future__ import annotations

from typing import Literal

import streamlit as st

from catcher_llm.schemas.chat import ChatMessage

CHAT_STATE_KEY = "chat_messages"
ChatRole = Literal["assistant", "system", "user"]


def init_chat_state() -> None:
    """Streamlit 세션 상태에 채팅 메시지 저장소를 초기화한다."""
    st.session_state.setdefault(CHAT_STATE_KEY, [])


def get_chat_messages() -> list[ChatMessage]:
    """현재 Streamlit 세션에 저장된 채팅 메시지 목록을 반환한다."""
    init_chat_state()
    return st.session_state[CHAT_STATE_KEY]


def add_chat_message(role: ChatRole, content: str) -> None:
    """지정한 역할과 본문으로 새 채팅 메시지를 세션 상태에 추가한다."""
    messages = get_chat_messages()
    messages.append(ChatMessage(role=role, content=content))


def reset_chat_state() -> None:
    """Streamlit 세션의 채팅 메시지 목록을 빈 상태로 되돌린다."""
    st.session_state[CHAT_STATE_KEY] = []
