from __future__ import annotations

from typing import Literal

import streamlit as st

from catcher_llm.schemas.chat import ChatMessage

CHAT_STATE_KEY = "chat_messages"
ChatRole = Literal["assistant", "system", "user"]


def init_chat_state() -> None:
    st.session_state.setdefault(CHAT_STATE_KEY, [])


def get_chat_messages() -> list[ChatMessage]:
    init_chat_state()
    return st.session_state[CHAT_STATE_KEY]


def add_chat_message(role: ChatRole, content: str) -> None:
    messages = get_chat_messages()
    messages.append(ChatMessage(role=role, content=content))


def reset_chat_state() -> None:
    st.session_state[CHAT_STATE_KEY] = []
