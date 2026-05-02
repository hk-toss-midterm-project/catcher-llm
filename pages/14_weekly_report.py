from __future__ import annotations

import runpy


def _run_weekly_report_page() -> None:
    """주간 리포트 래퍼 페이지에서 개발용 주간 리포트 스크립트를 실행한다."""
    runpy.run_path("dev_pages/14_weekly_report_rim.py", run_name="__main__")


_run_weekly_report_page()
