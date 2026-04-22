from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.chat import ChatMessage


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.title(settings.app_name)
        st.caption(f"Environment: {settings.env_name}")
        st.write(f"Model: {settings.openai_model}")
        st.write(f"Raw data: `{settings.raw_data_dir}`")
        st.write(f"Vector store: `{settings.vectorstore_dir}`")


def render_chat_messages(messages: Iterable[ChatMessage]) -> None:
    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)
