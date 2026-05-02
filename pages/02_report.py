from __future__ import annotations

from pathlib import Path

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


def _find_page_path(file_name: str) -> str | None:
    candidates = [
        Path("pages") / file_name,
        Path("dev_pages") / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path.as_posix()

    return None


def _go_to_report(file_name: str) -> None:
    page_path = _find_page_path(file_name)

    if page_path is None:
        st.error(
            f"페이지 파일을 찾을 수 없습니다: {file_name}\n\n"
            "pages 또는 dev_pages 폴더 안에 파일이 있는지 확인해 주세요."
        )
        return

    st.switch_page(page_path)


_ensure_logged_in()

st.title("📫 리포트 조회")
st.caption("원하는 리포트 유형을 누르면 해당 분석 페이지로 이동합니다.")

st.markdown("## 리포트 유형 선택")

daily_col, weekly_col, monthly_col = st.columns(3)

with daily_col:
    if st.button("일간 레포트", use_container_width=True):
        _go_to_report("13_daily_report_rim.py")

with weekly_col:
    if st.button("주간 레포트", use_container_width=True):
        _go_to_report("14_weekly_report_rim.py")

with monthly_col:
    if st.button("월간 레포트", use_container_width=True):
        _go_to_report("15_monthly_report_rim.py")

st.markdown("---")
st.info("리포트 페이지 안에서 날짜와 사용자 조건을 선택해 상세 결과를 확인할 수 있습니다.")