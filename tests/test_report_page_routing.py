from __future__ import annotations

from pathlib import Path


def test_report_page_routes_daily_weekly_monthly_buttons_to_wrapper_pages() -> None:
    """리포트 조회 페이지가 버튼별로 pages 디렉터리의 래퍼 리포트 페이지로 이동하는지 검증한다."""
    page_source = Path("pages/02_report.py").read_text(encoding="utf-8")

    assert 'st.switch_page("pages/13_daily_report.py")' in page_source
    assert 'st.switch_page("pages/14_weekly_report.py")' in page_source
    assert 'st.switch_page("pages/15_monthly_report.py")' in page_source
    assert "일간 레포트" in page_source
    assert "주간 레포트" in page_source
    assert "월간 레포트" in page_source


def test_report_wrapper_pages_delegate_to_matching_dev_pages() -> None:
    """래퍼 리포트 페이지가 대응하는 dev_pages 스크립트를 실행하도록 연결되는지 검증한다."""
    daily_wrapper = Path("pages/13_daily_report.py").read_text(encoding="utf-8")
    weekly_wrapper = Path("pages/14_weekly_report.py").read_text(encoding="utf-8")
    monthly_wrapper = Path("pages/15_monthly_report.py").read_text(encoding="utf-8")

    assert 'runpy.run_path("dev_pages/13_daily_report_rim.py"' in daily_wrapper
    assert 'runpy.run_path("dev_pages/14_weekly_report_rim.py"' in weekly_wrapper
    assert 'runpy.run_path("dev_pages/15_monthly_report_rim.py"' in monthly_wrapper
