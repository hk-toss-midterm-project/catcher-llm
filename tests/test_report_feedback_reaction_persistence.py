from __future__ import annotations

from pathlib import Path


def test_daily_weekly_monthly_report_pages_persist_feedback_reaction() -> None:
    """일·주·월 리포트 페이지가 피드백 투표를 세션 테이블에도 저장하는지 검증한다."""
    page_expectations = {
        "pages/04_daily_report.py": [
            "save_session_feedback_reaction",
            'period_type="daily"',
        ],
        "pages/05_weekly_report.py": [
            "save_session_feedback_reaction",
            'period_type="weekly"',
        ],
        "pages/06_monthly_report.py": [
            "save_session_feedback_reaction",
            'period_type="monthly"',
        ],
    }

    for page_path, snippets in page_expectations.items():
        page_source = Path(page_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in page_source


def test_daily_and_monthly_report_pages_collect_dislike_reason_like_weekly_page() -> None:
    """일일·월간 리포트도 주간 리포트처럼 아쉬운 점 입력 UI를 제공하는지 검증한다."""
    page_expectations = {
        "pages/04_daily_report.py": [
            "daily_like_btn_active",
            "daily_dislike_btn_active",
            "👎 아쉬워요",
            "daily_feedback_reason",
            "아쉬웠던 점",
            "의견 제출",
        ],
        "pages/06_monthly_report.py": [
            "monthly_like_btn_active",
            "monthly_dislike_btn_active",
            "👎 아쉬워요",
            "monthly_feedback_reason",
            "아쉬웠던 점",
            "의견 제출",
        ],
    }

    for page_path, snippets in page_expectations.items():
        page_source = Path(page_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in page_source

    daily_source = Path("pages/04_daily_report.py").read_text(encoding="utf-8")
    assert "👎 싫어요" not in daily_source
