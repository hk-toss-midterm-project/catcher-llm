from __future__ import annotations

from pathlib import Path

from catcher_llm.ui.report_auth import coerce_session_user_id


def test_report_page_routes_daily_weekly_monthly_buttons_to_wrapper_pages() -> None:
    """리포트 조회 페이지가 버튼별로 pages 디렉터리의 래퍼 리포트 페이지로 이동하는지 검증한다."""
    page_source = Path("pages/02_report.py").read_text(encoding="utf-8")

    assert 'st.switch_page("pages/04_daily_report.py")' in page_source
    assert 'st.switch_page("pages/05_weekly_report.py")' in page_source
    assert 'st.switch_page("pages/06_monthly_report.py")' in page_source
    assert "일일 리포트" in page_source
    assert "주간 리포트" in page_source
    assert "월간 리포트" in page_source


def test_report_detail_pages_are_implemented_in_pages_directory() -> None:
    """일일·주간·월간 상세 리포트가 dev 페이지 래퍼 없이 pages에서 직접 구현되는지 검증한다."""
    daily_page = Path("pages/04_daily_report.py").read_text(encoding="utf-8")
    weekly_page = Path("pages/05_weekly_report.py").read_text(encoding="utf-8")
    monthly_page = Path("pages/06_monthly_report.py").read_text(encoding="utf-8")

    assert "run_logged_in_report_page" not in daily_page
    assert "run_logged_in_report_page" not in weekly_page
    assert "run_logged_in_report_page" not in monthly_page
    assert "select_daily_date" in daily_page
    assert "select_week_range" in weekly_page
    assert "select_month" in monthly_page


def test_report_pages_use_consistent_top_date_picker_layout() -> None:
    """일일·주간·월간 리포트 상단에서 같은 제목·날짜 선택 레이아웃을 쓰는지 검증한다."""
    daily_page = Path("pages/04_daily_report.py").read_text(encoding="utf-8")
    weekly_page = Path("pages/05_weekly_report.py").read_text(encoding="utf-8")
    monthly_page = Path("pages/06_monthly_report.py").read_text(encoding="utf-8")

    assert "top1, top2 = st.columns([1.08, 1.12])" in daily_page
    assert "f2, f3 = st.columns([1.27, 1.08])" in daily_page
    assert "비교 기준일" not in daily_page
    assert "top1, top2 = st.columns([1.08, 1.12])" in weekly_page
    assert "f1, f2, f3 = st.columns([0.9, 1.32, 1.08])" in weekly_page
    assert "f2, f3 = st.columns([1.32, 1.08])" in weekly_page
    assert "top1, top2 = st.columns([1.08, 1.12])" in monthly_page
    assert "f1, f2, f3 = st.columns([0.9, 1.27, 1.08])" in monthly_page
    assert "f2, f3 = st.columns([1.27, 1.08])" in monthly_page


def test_report_pages_do_not_force_global_light_background() -> None:
    """일일·주간·월간 리포트가 다크 모드에서 전역 배경을 강제로 밝게 고정하지 않는지 검증한다."""
    page_paths = [
        Path("pages/04_daily_report.py"),
        Path("pages/05_weekly_report.py"),
        Path("pages/06_monthly_report.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert ".stApp { background:#f8fafc; }" not in page_source


def test_report_page_titles_follow_streamlit_theme_text_color() -> None:
    """일일·주간·월간 리포트 제목이 다크 모드에서 고정 어두운 색을 쓰지 않는지 검증한다."""
    page_paths = [
        Path("pages/04_daily_report.py"),
        Path("pages/05_weekly_report.py"),
        Path("pages/06_monthly_report.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        page_title_css = page_source.split(".page-title {", maxsplit=1)[1].split(
            "}",
            maxsplit=1,
        )[0]
        assert "color: var(--text-color)" in page_title_css
        assert "#0f172a" not in page_title_css


def test_app_registers_report_detail_pages_for_logged_in_navigation() -> None:
    """메인 앱이 로그인 사용자 내비게이션에 상세 리포트 페이지를 등록하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    expected_snippets = [
        "daily_report_pg = st.Page(",
        '"pages/04_daily_report.py"',
        "weekly_report_pg = st.Page(",
        '"pages/05_weekly_report.py"',
        "monthly_report_pg = st.Page(",
        '"pages/06_monthly_report.py"',
        "daily_report_pg",
        "weekly_report_pg",
        "monthly_report_pg",
    ]

    for snippet in expected_snippets:
        assert snippet in app_source

    assert 'title="일일 리포트"' in app_source


def test_report_pages_use_logged_in_user_id_when_session_exists() -> None:
    """리포트 dev 페이지가 로그인 세션 사용자 ID를 우선 사용하도록 연결됐는지 검증한다."""
    page_paths = [
        Path("dev_pages/13_daily_report_rim.py"),
        Path("dev_pages/14_weekly_report_rim.py"),
        Path("dev_pages/15_monthly_report_rim.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert "get_logged_in_user_id_from_session" in page_source
        assert "logged_in_member_id = get_logged_in_user_id_from_session()" in page_source
        assert "if logged_in_member_id is None:" in page_source


def test_report_session_user_id_coercion() -> None:
    """로그인 세션의 사용자 ID 후보를 리포트용 정수 ID로 정규화하는지 검증한다."""
    assert coerce_session_user_id(3) == 3
    assert coerce_session_user_id(" 4 ") == 4
    assert coerce_session_user_id(None) is None
    assert coerce_session_user_id("") is None
    assert coerce_session_user_id("abc") is None
    assert coerce_session_user_id(True) is None
