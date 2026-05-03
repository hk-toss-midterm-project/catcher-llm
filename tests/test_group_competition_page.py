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


def test_group_competition_page_aligns_action_card_heights_with_blank_line() -> None:
    """활동할 그룹 선택 카드가 고정 높이 대신 공백 줄로 그룹 만들기 카드와 높이를 맞추는지 검증한다."""
    page_source = Path("pages/03_group_competition.py").read_text(encoding="utf-8")

    assert "action-card" not in page_source
    assert "min-height: 124px;" not in page_source
    assert "선택한 그룹 기준으로 리더보드와 피드백 실천 현황이 바뀝니다.<br>&nbsp;" in page_source


def test_group_competition_page_tunes_spacing_and_help_button_shape() -> None:
    """그룹 경쟁 페이지가 리더보드/피드백 여백과 초대 도움말 버튼 크기를 조정하는지 검증한다."""
    page_source = Path("pages/03_group_competition.py").read_text(encoding="utf-8")

    expected_snippets = [
        ".leaderboard-item {",
        "margin-bottom: 10px;",
        ".feedback-panel {",
        "margin-bottom: 18px;",
        'button[aria-label="Help for 초대할 사용자 ID"]',
        "width: 28px;",
        "height: 28px;",
        "border-radius: 999px;",
    ]
    for snippet in expected_snippets:
        assert snippet in page_source
