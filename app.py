from __future__ import annotations

import streamlit as st

from catcher_llm.config.logging import configure_logging
from catcher_llm.config.settings import get_settings
from catcher_llm.ui.components import render_sidebar

configure_logging()
settings = get_settings()

st.set_page_config(page_title=settings.app_name, layout="wide")

render_sidebar(settings)

st.title(settings.app_name)
st.write("Streamlit handles the interface and LangChain stays inside the src package.")
st.info("Start with the Chat page, then wire document ingestion and retrieval on the Docs page.")
