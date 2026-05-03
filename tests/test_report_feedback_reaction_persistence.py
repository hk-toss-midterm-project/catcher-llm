from __future__ import annotations

from pathlib import Path


def test_daily_weekly_monthly_report_pages_persist_feedback_reaction() -> None:
    """일·주·월 리포트 페이지가 피드백 투표를 세션 테이블에도 저장하는지 검증한다."""
    page_expectations = {
        "dev_pages/13_daily_report_rim.py": [
            "save_session_feedback_reaction",
            'period_type="daily"',
        ],
        "dev_pages/14_weekly_report_rim.py": [
            "save_session_feedback_reaction",
            'period_type="weekly"',
        ],
        "dev_pages/15_monthly_report_rim.py": [
            "save_session_feedback_reaction",
            'period_type="monthly"',
        ],
    }

    for page_path, snippets in page_expectations.items():
        page_source = Path(page_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in page_source
