from __future__ import annotations

from pathlib import Path


def test_app_registers_group_competition_page_for_logged_in_users() -> None:
    """메인 앱이 로그인 사용자용 그룹 경쟁 페이지를 네비게이션에 연결하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert '"pages/03_group_competition.py"' in app_source
    assert 'title="그룹 경쟁"' in app_source
    assert "group_pg" in app_source


def test_group_competition_page_wires_group_service_flow() -> None:
    """그룹 경쟁 페이지가 그룹 생성, 즉시 랭킹, 피드백 실천 현황을 연결하는지 검증한다."""
    page_source = Path("pages/03_group_competition.py").read_text(encoding="utf-8")

    expected_snippets = [
        "create_group",
        "get_group_leaderboard_for_group",
        "get_group_member_feedback_status",
        "내 그룹",
        "그룹 만들기",
        "리더보드",
        "피드백 실천 현황",
    ]
    for snippet in expected_snippets:
        assert snippet in page_source
