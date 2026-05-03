from __future__ import annotations

from pathlib import Path


def test_app_profile_page_renders_personal_score_metric() -> None:
    """메인 프로필 페이지가 개인 점수 메트릭을 한국어 라벨로 노출하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert 'st.metric("개인 점수", profile.get("personal_score") or 0)' in app_source
    assert 'st.metric("personal_score"' not in app_source
    assert "refreshed_profile = authenticate_user(" in app_source


def test_authenticate_user_profile_includes_personal_score() -> None:
    """로그인 후 세션에 담기는 사용자 프로필이 personal_score를 포함하는지 검증한다."""
    service_source = Path("src/catcher_llm/services/user_data_service.py").read_text(
        encoding="utf-8"
    )

    assert '"personal_score": user.personal_score' in service_source


def test_app_persists_login_with_browser_cookie() -> None:
    """메인 앱이 새로고침 후에도 쿠키 기반 자동 로그인을 복원하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert "LOGIN_USER_ID_COOKIE" in app_source
    assert "LOGIN_USER_NAME_COOKIE" in app_source
    assert "def persist_login_cookie(" in app_source
    assert "document.cookie" in app_source
    assert "st.iframe(" in app_source
    assert "height=1" in app_source
    assert "height=0" not in app_source
    assert "components.html(" not in app_source
    assert "streamlit.components.v1" not in app_source
    assert "def restore_login_from_cookie(" in app_source
    assert "st.context.cookies" in app_source
    assert "persist_login_cookie(user_id, input_user_name.strip())" in app_source
    assert 'persist_login_cookie(int(st.session_state.user_id), str(profile["name"]))' in app_source
    assert "clear_login_cookie()" in app_source
    assert "restore_login_from_cookie()" in app_source


def test_app_logout_clears_cookie_before_cookie_restore() -> None:
    """로그아웃 요청이 쿠키 자동 복원보다 먼저 처리되는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert "def apply_pending_logout(" in app_source
    assert "logout_was_applied = apply_pending_logout()" in app_source
    assert "if not logout_was_applied:" in app_source
    assert 'st.button("로그아웃", width="stretch", on_click=logout)' in app_source
    logout_apply_index = app_source.index("logout_was_applied = apply_pending_logout()")
    restore_call_index = app_source.index(
        "restore_login_from_cookie()",
        app_source.index("if not logout_was_applied:"),
    )
    assert logout_apply_index < restore_call_index
