from __future__ import annotations

from catcher_llm.ui.report_auth import run_logged_in_report_page


def _run_monthly_report_page() -> None:
    """월간 리포트 래퍼 페이지에서 로그인 확인 후 개발용 리포트 스크립트를 실행한다."""
    run_logged_in_report_page("dev_pages/15_monthly_report_rim.py")


_run_monthly_report_page()
