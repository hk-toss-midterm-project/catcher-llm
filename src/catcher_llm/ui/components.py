from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

from catcher_llm.config.settings import Settings
from catcher_llm.schemas.chat import ChatMessage

_READONLY_CONTROL_CSS = """
<style>
.catcher-readonly-control-label {
    margin-bottom: 0.25rem;
    color: rgb(49, 51, 63);
    font-size: 0.875rem;
    line-height: 1.25rem;
}

.catcher-readonly-control-box {
    display: flex;
    align-items: center;
    width: 100%;
    min-height: 38px;
    padding: 0.25rem 0.75rem;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    background-color: rgb(240, 242, 246);
    color: rgb(49, 51, 63);
    font-size: 1rem;
    line-height: 1.5rem;
}
</style>
"""


def render_readonly_control(label: str, value: object) -> None:
    """폼 컨트롤 행에서 읽기 전용 값을 입력 박스와 같은 형태로 표시한다."""
    safe_label = escape(label)
    safe_value = escape(str(value))
    st.markdown(
        f"{_READONLY_CONTROL_CSS}"
        f'<div class="catcher-readonly-control-label">{safe_label}</div>'
        f'<div class="catcher-readonly-control-box">{safe_value}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(settings: Settings) -> None:
    """앱 이름, 환경, 모델, 데이터 경로 정보를 Streamlit 사이드바에 표시한다."""
    with st.sidebar:
        st.title(settings.app_name)
        st.caption(f"Environment: {settings.env_name}")
        st.write(f"Chat model: {settings.chat_model_label}")
        st.write(f"Embedding model: {settings.embedding_model_label}")
        st.write(f"Raw data: `{settings.raw_data_dir}`")
        st.write(f"Vector store: `{settings.vectorstore_dir}`")


def render_chat_messages(messages: Iterable[ChatMessage]) -> None:
    """저장된 채팅 메시지들을 Streamlit 채팅 UI에 순서대로 렌더링한다."""
    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)
