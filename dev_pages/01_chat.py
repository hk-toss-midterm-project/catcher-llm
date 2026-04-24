from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import configure_langsmith_env, get_settings
from catcher_llm.services.chat_service import generate_reply
from catcher_llm.ui.components import render_chat_messages, render_sidebar
from catcher_llm.ui.state import (
    add_chat_message,
    get_chat_messages,
    init_chat_state,
    reset_chat_state,
)

settings = get_settings()
configure_langsmith_env(settings)

render_sidebar(settings)
init_chat_state()

st.title("Chat")
st.caption(
    "The page stays thin. Use docs, document, rag, 문서, or 검색 in the prompt to route through retrieval."
)

controls, _ = st.columns([1, 4])
with controls:
    if st.button("Reset chat", use_container_width=True):
        reset_chat_state()
        st.rerun()

messages = get_chat_messages()
render_chat_messages(messages)

prompt = st.chat_input("Ask a question")
if prompt:
    add_chat_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    history = get_chat_messages()[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            result = generate_reply(prompt, history=history, settings=settings)
            st.write(result.reply)

    add_chat_message("assistant", result.reply)
