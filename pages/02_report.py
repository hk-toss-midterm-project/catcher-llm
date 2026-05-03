from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="리포트 조회", page_icon="📫", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _ensure_logged_in() -> None:
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("먼저 로그인해 주세요.")
        st.stop()


_ensure_logged_in()

st.title("📫 리포트 조회")
st.caption("원하는 리포트 유형을 누르면 해당 분석 페이지로 이동합니다.")

st.markdown("## 리포트 유형 선택")

daily_col, weekly_col, monthly_col = st.columns(3)

with daily_col:
    if st.button("일일 리포트", width="stretch"):
        st.switch_page("pages/04_daily_report.py")

with weekly_col:
    if st.button("주간 리포트", width="stretch"):
        st.switch_page("pages/05_weekly_report.py")

with monthly_col:
    if st.button("월간 리포트", width="stretch"):
        st.switch_page("pages/06_monthly_report.py")

st.markdown("---")
st.info("리포트 페이지 안에서 날짜와 사용자 조건을 선택해 상세 결과를 확인할 수 있습니다.")
