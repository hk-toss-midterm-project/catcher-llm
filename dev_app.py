from __future__ import annotations

import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.services.ingestion_service import ensure_feedback_vectorstores
from catcher_llm.services.user_data_service import ensure_user_database
from catcher_llm.ui.dev_navigation import get_dev_page_specs

st.set_page_config(
    page_title="Catcher Dev",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_startup_resources() -> None:
    """개발 앱 시작 시 사용자 DB와 피드백 벡터스토어를 한 번 초기화한다."""
    ensure_user_database(settings=get_settings())
    ensure_feedback_vectorstores(settings=get_settings())


init_startup_resources()

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
