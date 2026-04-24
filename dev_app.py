from __future__ import annotations

import streamlit as st

from catcher_llm.ui.dev_navigation import get_dev_page_specs

st.set_page_config(
    page_title="Catcher Dev",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page(
            str(spec.path),
            title=spec.title,
            icon=spec.icon,
            default=spec.default,
        )
        for spec in get_dev_page_specs()
    ]
)
navigation.run()
