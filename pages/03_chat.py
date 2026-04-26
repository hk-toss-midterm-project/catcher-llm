from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.services.chat_service import generate_reply
from catcher_llm.services.session_history_service import (
    build_user_session_id,
    clear_session_chat_messages,
    load_session_chat_messages,
)
from catcher_llm.ui.components import render_chat_messages
from catcher_llm.ui.state import (
    CHAT_STATE_KEY,
    add_chat_message,
    get_chat_messages,
    init_chat_state,
    reset_chat_state,
)

LANGCHAIN_SESSION_ID_KEY = "langchain_session_id"
CHAT_STATE_LOADED_KEY = "chat_state_loaded_from_sqlite"

settings = get_settings()
configure_langsmith_env(settings)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("먼저 로그인해주세요.")
    st.stop()

user_id_value = st.session_state.get("user_id")
if not isinstance(user_id_value, int):
    st.warning("사용자 세션을 확인할 수 없습니다. 다시 로그인해주세요.")
    st.stop()

session_id_value = st.session_state.get(LANGCHAIN_SESSION_ID_KEY)
if isinstance(session_id_value, str) and session_id_value.strip():
    session_id = session_id_value
else:
    session_id = build_user_session_id(user_id_value)
    st.session_state[LANGCHAIN_SESSION_ID_KEY] = session_id

init_chat_state()
if not bool(st.session_state.get(CHAT_STATE_LOADED_KEY, False)):
    st.session_state[CHAT_STATE_KEY] = load_session_chat_messages(session_id, settings=settings)
    st.session_state[CHAT_STATE_LOADED_KEY] = True

st.title("AI 상담")

controls, _ = st.columns([1, 4])
with controls:
    if st.button("대화 초기화", use_container_width=True):
        clear_session_chat_messages(session_id, settings=settings)
        reset_chat_state()
        st.session_state[CHAT_STATE_LOADED_KEY] = True
        st.rerun()

messages = get_chat_messages()
render_chat_messages(messages)

prompt = st.chat_input("소비 상담 질문을 입력하세요")
if prompt:
    add_chat_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    history = get_chat_messages()[:-1]
    with st.chat_message("assistant"):
        with st.spinner("응답 생성 중..."):
            result = generate_reply(
                prompt,
                history=history,
                settings=settings,
                session_id=session_id,
            )
            st.write(result.reply)

    add_chat_message("assistant", result.reply)
