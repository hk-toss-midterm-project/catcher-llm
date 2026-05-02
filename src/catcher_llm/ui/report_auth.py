from __future__ import annotations

import runpy

import streamlit as st


def coerce_session_user_id(raw_user_id: object) -> int | None:
    """세션에 저장된 사용자 ID 후보를 리포트 조회용 정수 ID로 변환한다."""
    if raw_user_id is None or isinstance(raw_user_id, bool):
        return None

    if isinstance(raw_user_id, int):
        return raw_user_id

    if isinstance(raw_user_id, str):
        stripped_user_id = raw_user_id.strip()
        if stripped_user_id == "":
            return None

        try:
            return int(stripped_user_id)
        except ValueError:
            return None

    return None


def get_logged_in_user_id_from_session() -> int | None:
    """로그인된 Streamlit 세션의 사용자 ID를 반환하고 개발 실행이면 None을 반환한다."""
    logged_in: object = st.session_state.get("logged_in", False)
    if logged_in is not True:
        return None

    raw_user_id: object = st.session_state.get("user_id")
    user_id = coerce_session_user_id(raw_user_id)
    if user_id is None:
        st.error("로그인한 사용자 ID를 찾을 수 없습니다.")
        st.stop()

    return user_id


def require_logged_in_user_id() -> int:
    """pages 상세 리포트 접근 전에 로그인 사용자 ID가 있는지 확인한다."""
    user_id = get_logged_in_user_id_from_session()
    if user_id is None:
        st.warning("먼저 로그인해 주세요.")
        st.stop()

    return user_id


def run_logged_in_report_page(script_path: str) -> None:
    """로그인 세션을 확인한 뒤 지정한 리포트 스크립트를 현재 페이지에서 실행한다."""
    require_logged_in_user_id()
    runpy.run_path(script_path, run_name="__main__")
