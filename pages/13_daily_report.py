from __future__ import annotations

import runpy


def _run_daily_report_page() -> None:
    """일간 리포트 래퍼 페이지에서 개발용 일간 리포트 스크립트를 실행한다."""
    runpy.run_path("dev_pages/13_daily_report_rim.py", run_name="__main__")


_run_daily_report_page()
