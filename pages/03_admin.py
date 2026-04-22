from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.ui.components import render_sidebar
from catcher_llm.ui.state import reset_chat_state

settings = get_settings()

st.set_page_config(page_title=f"{settings.app_name} | Admin", layout="wide")

render_sidebar(settings)

st.title("Admin")
st.caption("Use this page for runtime checks, environment review, and local reset actions.")

if st.button("Clear chat session"):
    reset_chat_state()
    st.success("Session state cleared.")

st.subheader("Settings")
st.json(
    {
        "app_name": settings.app_name,
        "env_name": settings.env_name,
        "openai_model": settings.openai_model,
        "embedding_model": settings.embedding_model,
        "has_openai_key": settings.has_openai_key,
        "has_langsmith_key": settings.has_langsmith_key,
        "langsmith_project": settings.langsmith_project,
        "langsmith_dataset_name": settings.langsmith_dataset_name,
        "raw_data_dir": str(settings.raw_data_dir),
        "processed_data_dir": str(settings.processed_data_dir),
        "vectorstore_dir": str(settings.vectorstore_dir),
        "eval_data_dir": str(settings.eval_data_dir),
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
        "rag_top_k": settings.rag_top_k,
    }
)
