from __future__ import annotations

from pathlib import Path


def test_app_registers_group_competition_page_for_logged_in_users() -> None:
    """메인 앱이 로그인 사용자용 그룹 경쟁 페이지를 내비게이션에 연결하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert '"pages/03_group_competition.py"' in app_source
    assert 'title="그룹 경쟁"' in app_source
    assert "group_pg" in app_source


def test_group_competition_page_wires_group_service_flow() -> None:
    """그룹 경쟁 페이지가 생성·대회·공유·리더보드 서비스를 모두 연결하는지 검증한다."""
    page_source = Path("pages/03_group_competition.py").read_text(encoding="utf-8")

    expected_snippets = [
        "create_group",
        "create_competition",
        "share_transaction_to_group",
        "get_group_feed",
        "get_group_leaderboard",
        "내 그룹",
        "그룹 만들기",
        "대회 만들기",
        "소비 공유",
        "리더보드",
    ]
    for snippet in expected_snippets:
        assert snippet in page_source
