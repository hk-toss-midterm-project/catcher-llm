from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.chat import ChatMessage


def render_sidebar(settings: Settings) -> None:
    """앱 이름, 환경, 모델, 데이터 경로 정보를 Streamlit 사이드바에 표시한다."""
    with st.sidebar:
        st.title(settings.app_name)
        st.caption(f"Environment: {settings.env_name}")
        st.write(f"Model: {settings.openai_model}")
        st.write(f"Raw data: `{settings.raw_data_dir}`")
        st.write(f"Vector store: `{settings.vectorstore_dir}`")


def render_chat_messages(messages: Iterable[ChatMessage]) -> None:
    """저장된 채팅 메시지들을 Streamlit 채팅 UI에 순서대로 렌더링한다."""
    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)
