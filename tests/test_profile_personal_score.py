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
