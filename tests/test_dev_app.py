from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from catcher_llm.db.models import SessionModel
from catcher_llm.schemas.consumption_feedback import (
    DailyFeedbackMemoryContext,
    DailyFeedbackResult,
    DailyFeedbackServiceResult,
    DailyFeedbackSessionContext,
    MonthlyFeedbackResult,
    MonthlyFeedbackServiceResult,
    UserProfileContext,
    UserSpendingData,
    WeeklyFeedbackResult,
    WeeklyFeedbackServiceResult,
)
from catcher_llm.services.consumption_feedback.daily_feedback import DailyFeedbackTimingRecord
from catcher_llm.services.consumption_feedback.feedback_reaction import (
    FeedbackReactionSaveResult,
)
from catcher_llm.ui.dev_navigation import get_dev_page_specs, get_repo_root


def _click_button_by_label(app: AppTest, label: str) -> None:
    """Streamlit 테스트 앱에서 지정한 라벨의 버튼을 찾아 클릭한다."""
    for button in app.button:
        if button.label == label:
            button.click().run(timeout=10)
            return
    msg = f"버튼을 찾을 수 없습니다: {label}"
    raise AssertionError(msg)


def _make_daily_analysis_result() -> dict[str, object]:
    """일일 분석 개발 페이지 테스트에 사용할 새 지표 포함 결과를 만든다."""
    return {
        "member_id": 1,
        "analysis_date": "2024-04-01",
        "source_paths": {"past_source": "sqlite", "today_source": "sqlite"},
        "outlier_thresholds": {
            "q1": 1000.0,
            "q3": 3000.0,
            "iqr": 2000.0,
            "lower_bound": 0.0,
            "upper_bound": 6000.0,
        },
        "stable_metrics": {
            "past_daily_stable_average": 2000.0,
            "today_total": 133044,
            "increase_rate_percent": 10.5,
            "category_ratio_changes": [],
        },
        "anomaly_detection": {
            "past_daily_original_average": 2100.0,
            "spike_ratio": 1.1,
            "is_spike": False,
            "high_spending_threshold": 6000.0,
            "high_spending_items": [],
        },
        "previous_day_comparison": {
            "yesterday_date": "2024-03-31",
            "yesterday_total": 10000,
            "today_total": 133044,
            "amount_diff": 123044,
            "amount_diff_rate_percent": 1230.44,
            "yesterday_count": 2,
            "today_count": 9,
            "count_diff": 7,
            "yesterday_main_category": "식비",
            "today_main_category": "생활",
        },
        "daily_comparisons": {
            "previous_day": {
                "label": "어제 대비",
                "reference_date": "2024-03-31",
                "reference_total": 10000,
                "today_total": 133044,
                "amount_diff": 123044,
                "amount_diff_rate_percent": 1230.44,
                "reference_count": 2,
                "today_count": 9,
                "count_diff": 7,
                "reference_main_category": "식비",
                "today_main_category": "생활",
            },
            "same_weekday_last_week": {
                "label": "지난주 같은 요일 대비",
                "reference_date": "2024-03-25",
                "reference_total": 30000,
                "today_total": 133044,
                "amount_diff": 103044,
                "amount_diff_rate_percent": 343.48,
                "reference_count": 3,
                "today_count": 9,
                "count_diff": 6,
                "reference_main_category": "식비",
                "today_main_category": "생활",
            },
            "recent_4week_same_weekday_average": {
                "label": "최근 4주 같은 요일 평균 대비",
                "reference_dates": ["2024-03-25", "2024-03-18"],
                "reference_days": [
                    {"date": "2024-03-25", "total": 30000, "transaction_count": 3},
                    {"date": "2024-03-18", "total": 50000, "transaction_count": 4},
                ],
                "reference_day_count": 2,
                "average_total": 40000.0,
                "today_total": 133044,
                "amount_diff": 93044.0,
                "amount_diff_rate_percent": 232.61,
                "average_count": 3.5,
                "today_count": 9,
                "count_diff": 5.5,
                "today_main_category": "생활",
            },
        },
        "time_slot_analysis": {
            "peak_slot": "2.오전(06-11)",
            "time_slots": [],
        },
        "payment_behavior_analysis": {
            "frictionless_spending": {
                "keywords": ["온라인", "간편결제", "앱결제", "배달"],
                "transaction_count": 1,
                "total_amount": 1486,
                "ratio_percent": 1.1169,
            },
            "transaction_density": {
                "transaction_count": 9,
                "average_amount_per_transaction": 14782.6667,
            },
        },
        "daily_metrics": {
            "daily_total_amount": 133044,
            "daily_transaction_count": 9,
            "daily_average_transaction_amount": 14782.6667,
            "daily_max_transaction_amount": 50000,
            "late_night_ratio_percent": 0.0,
            "daily_budget_usage_rate_percent": None,
            "no_spending_day": False,
            "daily_anomaly_score": 1.1,
            "time_slot_amounts": [],
            "category_spending": [],
            "special_metrics": {
                "impulse_spending_score": 0.2,
                "daily_spending_risk": 1.1,
            },
        },
    }


def test_get_repo_root_points_to_project_root() -> None:
    """개발 앱 네비게이션이 저장소 루트를 올바르게 기준점으로 삼는지 검증한다."""
    root = get_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "dev_pages").exists()


def test_get_dev_page_specs_registers_dev_pages_directory() -> None:
    """개발용 Streamlit 페이지들이 사이드바 표시 순서와 메타데이터로 등록되는지 검증한다."""
    specs = get_dev_page_specs()

    assert [spec.path.name for spec in specs] == [
        "01_chat.py",
        "02_retriever_probe.py",
        "03_document_rag.py",
        "04_daily_analysis.py",
        "05_daily_interpretation.py",
        "06_daily_feedback.py",
        "07_weekly_analysis.py",
        "08_weekly_interpretation.py",
        "09_weekly_feedback.py",
        "10_monthly_analysis.py",
        "11_monthly_interpretation.py",
        "12_monthly_feedback.py",
        "13_daily_report_rim.py",
        "14_weekly_report_rim.py",
        "15_monthly_report_rim.py",
        "16_user_trend_report.py",
        "17_daily_feedback_timing.py",
        "18_daily_interpretation_compare.py",
        "19_daily_feedback_unified.py",
        "20_model_comparison.py",
    ]
    assert [spec.title for spec in specs] == [
        "Chat",
        "Retriever 테스트",
        "문서 RAG",
        "일일 소비 분석",
        "일일 소비 해석 체인",
        "일일 피드백",
        "주간 소비 분석",
        "주간 소비 해석 체인",
        "주간 피드백",
        "월간 소비 분석",
        "월간 소비 해석 체인",
        "월간 피드백",
        "일일 리포트",
        "주간 보고서",
        "월간 보고서",
        "사용자 동향 보고서 생성",
        "일일 피드백 소요 시간",
        "일일 해석 방식 비교",
        "일일 통합 피드백",
        "GPT 모델 성능 비교",
    ]
    assert [spec.icon for spec in specs] == [
        "💬",
        "🔎",
        "📄",
        "📊",
        "🧭",
        "📣",
        "🗓️",
        "🧭",
        "🧾",
        "📈",
        "🧭",
        "🧾",
        "📝",
        "🗓️",
        "📈",
        "📑",
        "⏱️",
        "🧪",
        "📣",
        "🔬",
    ]
    assert [spec.default for spec in specs] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(spec.path.is_file() for spec in specs)


def test_user_trend_report_dev_page_wires_metric_and_report_scripts() -> None:
    """사용자 동향 보고서 개발 페이지가 지표·보고서 생성 스크립트를 호출하는지 검증한다."""
    page_source = Path("dev_pages/16_user_trend_report.py").read_text(encoding="utf-8")

    assert "build_and_save_trend_metrics" in page_source
    assert "generate_latest_user_trend_report" in page_source
    assert "resolve_metrics_dir" in page_source
    assert "지표 생성" in page_source
    assert "보고서 생성" in page_source
    assert "start_date" in page_source
    assert "end_date" in page_source


def test_rag_dev_pages_include_feedback_document_kind_search() -> None:
    """RAG 개발 페이지들이 문서 종류별 피드백 RAG 검색 모드를 제공하는지 검증한다."""
    retriever_source = Path("dev_pages/02_retriever_probe.py").read_text(encoding="utf-8")
    document_rag_source = Path("dev_pages/03_document_rag.py").read_text(encoding="utf-8")

    for page_source in (retriever_source, document_rag_source):
        assert "retrieve_feedback_contexts" in page_source
        assert "DocumentKind" in page_source
        assert "문서 종류별 피드백 RAG" in page_source
        assert "usefulness_threshold" in page_source
        assert "document_kinds" in page_source
        assert "catcher_consumption_benchmark" in page_source
        assert "Catcher 2018.07-12 소비 벤치마크" in page_source
        assert "self_report" not in page_source

    assert "PDF 코퍼스 직접 검색" in retriever_source
    assert "단일 문서 RAG" in document_rag_source


def test_dev_app_renders_default_page_without_exception() -> None:
    """Streamlit 개발 앱의 기본 페이지가 import 예외 없이 렌더링되는지 검증한다."""
    app = AppTest.from_file("dev_app.py")

    app.run(timeout=10)

    assert len(app.exception) == 0


def test_dev_app_initializes_startup_resources() -> None:
    """개발 앱도 실행 시작 시 사용자 DB와 피드백 벡터스토어를 준비하는지 검증한다."""
    app_source = Path("dev_app.py").read_text(encoding="utf-8")

    assert "from catcher_llm.config.settings import get_settings" in app_source
    assert (
        "from catcher_llm.services.ingestion_service import ensure_feedback_vectorstores"
        in app_source
    )
    assert "from catcher_llm.services.user_data_service import ensure_user_database" in app_source
    assert "@st.cache_resource" in app_source
    assert "def init_startup_resources() -> None:" in app_source
    assert "ensure_user_database(settings=get_settings())" in app_source
    assert "ensure_feedback_vectorstores(settings=get_settings())" in app_source
    assert app_source.index("init_startup_resources()") < app_source.index(
        "navigation = st.navigation("
    )


def test_main_app_initializes_feedback_vectorstores_on_startup() -> None:
    """메인 앱도 첫 실행 시 피드백 벡터스토어 준비 함수를 호출하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert (
        "from catcher_llm.services.ingestion_service import ensure_feedback_vectorstores"
        in app_source
    )
    assert "def init_startup_resources() -> None:" in app_source
    assert "ensure_user_database(settings=settings)" in app_source
    assert "ensure_feedback_vectorstores(settings=settings)" in app_source
    assert app_source.index("init_startup_resources()") < app_source.index(
        'if "logged_in" not in st.session_state:'
    )


def test_daily_report_page_uses_global_calendar_default_date() -> None:
    """일일 리포트 페이지가 SQLite 탐색 날짜가 아니라 공통 달력 기본일을 쓰는지 검증한다."""
    page_source = Path("pages/04_daily_report.py").read_text(encoding="utf-8")

    assert "DEFAULT_CALENDAR_DATE" in page_source
    assert "default=DEFAULT_CALENDAR_DATE" in page_source
    assert 'value="1"' not in page_source
    assert "date(2024, 3, 31)" not in page_source
    assert "st.number_input(" in page_source
    assert '"Member ID"' in page_source
    assert "dev_report_member_id_input_enabled" in page_source


def test_daily_report_page_uses_configured_sqlite_for_points() -> None:
    """일일 리포트 포인트 조회와 갱신이 설정 SQLite 경로와 v3 점수 컬럼을 쓰는지 검증한다."""
    page_source = Path("pages/04_daily_report.py").read_text(encoding="utf-8")

    assert "get_settings().sqlite_db_path" in page_source
    assert "personal_score" in page_source
    assert r"C:\Users\user\catcher" not in page_source
    assert '"개인 점수"' not in page_source


def test_report_pages_remove_duplicated_derived_date_controls() -> None:
    """리포트 페이지가 기준 날짜 선택에서 계산 가능한 보조 날짜 UI를 표시하지 않는지 검증한다."""
    daily_report_source = Path("pages/04_daily_report.py").read_text(encoding="utf-8")
    weekly_report_source = Path("pages/05_weekly_report.py").read_text(encoding="utf-8")

    assert "전일 기준일" not in daily_report_source
    assert "default_selection.previous_date" not in daily_report_source
    assert "previous_date = analysis_date - timedelta(days=1)" in daily_report_source

    assert "select_week_range" in weekly_report_source
    assert "render_readonly_control" not in weekly_report_source
    assert "분석 종료일" not in weekly_report_source


def test_consumption_dev_pages_use_2026_april_calendar_defaults() -> None:
    """소비 개발 페이지의 일·주·월 기본 달력값이 2026-04-01 기준인지 검증한다."""
    from datetime import date

    from catcher_llm.ui.date_picker import (
        DEFAULT_CALENDAR_DATE,
        DEFAULT_CALENDAR_MONTH,
        _coerce_week_range,
    )

    assert DEFAULT_CALENDAR_DATE == date(2026, 4, 1)
    assert DEFAULT_CALENDAR_MONTH == "2026-04"
    assert _coerce_week_range(DEFAULT_CALENDAR_DATE, DEFAULT_CALENDAR_DATE) == (
        date(2026, 3, 29),
        date(2026, 4, 4),
    )

    daily_analysis_source = Path("dev_pages/04_daily_analysis.py").read_text(encoding="utf-8")
    assert "previous_day = analysis_day - timedelta(days=1)" in daily_analysis_source
    assert "전일 비교 기준일" not in daily_analysis_source

    daily_or_weekly_pages = [
        Path("dev_pages/04_daily_analysis.py"),
        Path("dev_pages/05_daily_interpretation.py"),
        Path("dev_pages/06_daily_feedback.py"),
        Path("dev_pages/07_weekly_analysis.py"),
        Path("dev_pages/08_weekly_interpretation.py"),
        Path("dev_pages/09_weekly_feedback.py"),
        Path("pages/04_daily_report.py"),
        Path("pages/05_weekly_report.py"),
    ]
    for page_path in daily_or_weekly_pages:
        page_source = page_path.read_text(encoding="utf-8")
        assert "DEFAULT_CALENDAR_DATE" in page_source
        assert "date(2024" not in page_source

    monthly_pages = [
        Path("dev_pages/10_monthly_analysis.py"),
        Path("dev_pages/11_monthly_interpretation.py"),
        Path("dev_pages/12_monthly_feedback.py"),
        Path("pages/06_monthly_report.py"),
    ]
    for page_path in monthly_pages:
        page_source = page_path.read_text(encoding="utf-8")
        assert "DEFAULT_CALENDAR_MONTH" in page_source
        assert '"2024-04"' not in page_source
        assert '"2024-03"' not in page_source


def test_consumption_dev_pages_use_popover_date_picker_helper() -> None:
    """일일·주간·월간 소비 페이지가 popover 날짜 선택 헬퍼로 날짜를 선택하는지 검증한다."""
    page_paths = [
        Path("dev_pages/04_daily_analysis.py"),
        Path("dev_pages/05_daily_interpretation.py"),
        Path("dev_pages/06_daily_feedback.py"),
        Path("dev_pages/07_weekly_analysis.py"),
        Path("dev_pages/08_weekly_interpretation.py"),
        Path("dev_pages/09_weekly_feedback.py"),
        Path("dev_pages/10_monthly_analysis.py"),
        Path("dev_pages/11_monthly_interpretation.py"),
        Path("dev_pages/12_monthly_feedback.py"),
        Path("pages/04_daily_report.py"),
        Path("pages/05_weekly_report.py"),
        Path("pages/06_monthly_report.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert "catcher_llm.ui.date_picker" in page_source
        assert "render_date_picker_styles()" in page_source
        assert ".date_input(" not in page_source
        assert 'text_input("분석 월"' not in page_source


def test_date_picker_helper_uses_streamlit_date_picker_popover() -> None:
    """날짜 선택 헬퍼가 streamlit-date-picker와 st.popover 기반으로 동작하는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert "streamlit_date_picker" in helper_source
    assert "st.popover" in helper_source
    assert "@st.dialog" not in helper_source
    assert "_DATE_PICKER_POPOVER_CSS" in helper_source
    assert 'div[data-baseweb="popover"]' in helper_source
    assert 'div[style*="height: 50px"]' in helper_source
    assert "_ensure_date_picker_bundle_height" in helper_source
    assert "min-height: 400px" in helper_source
    assert "height: 400px" in helper_source
    assert "--catcher-date-picker-content-width: 360px" in helper_source
    assert "--catcher-date-picker-popover-width: 350px" in helper_source
    assert "width: 350px" in helper_source
    assert "min-width: 350px" in helper_source
    assert "max-width: 350px" in helper_source
    assert "padding: 16px" in helper_source
    assert "box-sizing: border-box" in helper_source
    assert 'div[data-testid="stPopoverBody"] > div' in helper_source
    assert "width: 100%" in helper_source
    assert "min-width: 0" in helper_source
    assert (
        'div[data-baseweb="popover"] > div:has(> div[data-testid="stVerticalBlock"])'
        in helper_source
    )
    assert 'div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"]' in helper_source
    assert 'div[data-testid="stPopoverBody"] div[data-testid="stElementContainer"]' in helper_source
    assert "max-width: 100%" in helper_source
    assert "overflow: hidden" in helper_source
    assert "PickerType.week" in helper_source
    assert "PickerType.month" in helper_source
    assert "streamlit_calendar" not in helper_source


def test_date_picker_value_coercion() -> None:
    """streamlit-date-picker 반환값을 일·주·월 선택값으로 일관되게 변환하는지 검증한다."""
    from datetime import date, datetime

    from streamlit_date_picker import PickerType

    from catcher_llm.ui.date_picker import (
        _coerce_date,
        _coerce_month,
        _coerce_picker_date,
        _coerce_week_range,
    )

    default = date(2024, 4, 1)

    assert _coerce_date(datetime(2024, 4, 3, 12, 0), default) == date(2024, 4, 3)
    assert _coerce_date("2024-04-03", default) == date(2024, 4, 3)
    assert _coerce_week_range("2024-04-03", default) == (
        date(2024, 3, 31),
        date(2024, 4, 6),
    )
    assert _coerce_week_range("2024-14th", default) == (
        date(2024, 3, 31),
        date(2024, 4, 6),
    )
    assert _coerce_month("2024-04-18", default) == "2024-04"
    assert _coerce_month("2024-04", default) == "2024-04"
    assert _coerce_picker_date(PickerType.week, "2024-14th", default) == date(2024, 3, 31)
    assert _coerce_picker_date(PickerType.month, "2024-04", default) == date(2024, 4, 1)


def test_date_picker_pending_state_application() -> None:
    """picker pending 값을 적용하면 확정 선택값 세션 키에 저장되는지 검증한다."""
    from datetime import date

    from streamlit_date_picker import PickerType

    from catcher_llm.ui.date_picker import (
        _apply_pending_state_date,
        _pending_session_key,
        _popover_session_key,
        _session_key,
        _set_pending_state_date,
        _update_pending_state_from_picker,
        _update_popover_close_token,
    )

    state: dict[str, object] = {}

    _set_pending_state_date(state, "sample", date(2024, 4, 8))
    pending_date = _update_pending_state_from_picker(
        state,
        "sample",
        PickerType.date,
        None,
        date(2024, 4, 1),
    )
    applied_date = _apply_pending_state_date(state, "sample", date(2024, 4, 1))

    assert pending_date == date(2024, 4, 8)
    assert applied_date == date(2024, 4, 8)
    assert state[_session_key("sample")] == "2024-04-08"
    assert state[_pending_session_key("sample")] == "2024-04-08"

    popover_token = _update_popover_close_token(state, "sample")

    assert popover_token == 1
    assert state[_popover_session_key("sample")] == 1


def test_date_picker_popover_trigger_uses_display_value_box() -> None:
    """날짜 표시 박스 자체가 popover 트리거이고 별도 선택 버튼을 쓰지 않는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert "st.text_input(" not in helper_source
    assert "st.popover(\n        value_label," in helper_source
    assert '"선택",' not in helper_source
    assert "_DATE_PICKER_TRIGGER_CSS" in helper_source


def test_date_picker_trigger_uses_theme_aware_input_style() -> None:
    """날짜 박스가 다크 모드에 맞는 테마 변수와 입력 컨테이너형 스타일을 쓰는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert 'button[data-testid="stPopoverButton"]' in helper_source
    assert 'div[aria-haspopup="true"] > button[data-testid="stPopoverButton"]' in helper_source
    assert "var(--secondary-background-color)" in helper_source
    assert "var(--text-color)" in helper_source
    assert "color-mix(in srgb, var(--text-color) 20%, transparent)" in helper_source
    assert "background-color: var(--secondary-background-color) !important" in helper_source
    assert "border: 1px solid color-mix" in helper_source
    assert "border-radius: 0.5rem !important" in helper_source
    assert "min-height: 2.5rem !important" in helper_source
    assert "box-shadow: none !important" in helper_source
    assert "font-weight: 400 !important" in helper_source
    assert "rgb(240, 242, 246)" not in helper_source
    assert "rgb(49, 51, 63)" not in helper_source
    assert "rgba(49, 51, 63" not in helper_source


def test_date_picker_does_not_add_extra_vertical_elements_before_popover() -> None:
    """날짜 선택기 타이틀은 widget label HTML로 유지하고 트리거 안에는 스타일 요소를 만들지 않는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert "st.caption(label)" not in helper_source
    assert "_render_picker_label(label)" in helper_source
    assert "render_date_picker_styles" in helper_source
    assert "st.markdown(_DATE_PICKER_TRIGGER_CSS, unsafe_allow_html=True)" in helper_source
    render_source = helper_source.split("def _render_picker_popover", maxsplit=1)[1]
    assert "_DATE_PICKER_TRIGGER_CSS" not in render_source
    assert "st.html(_DATE_PICKER_POPOVER_CSS)" in helper_source


def test_date_picker_title_uses_widget_label_markup() -> None:
    """날짜 선택기 타이틀이 Streamlit widget label과 같은 구조의 HTML을 쓰는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert 'data-testid="stWidgetLabel"' in helper_source
    assert 'data-testid="stMarkdownContainer"' in helper_source
    assert 'aria-hidden="true"' in helper_source
    assert "margin-bottom: 4px" in helper_source
    assert "color: var(--text-color) !important" in helper_source
    assert "stCaptionContainer" not in helper_source


def test_date_picker_label_and_box_gap_matches_widget_spacing() -> None:
    """날짜 선택기 label wrapper와 popover wrapper 사이 간격을 4px로 줄이는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert 'div[data-testid="stVerticalBlock"]:has(' in helper_source
    assert 'label[data-testid="stWidgetLabel"].catcher-date-picker-label' in helper_source
    assert 'div[data-testid="stLayoutWrapper"] div[data-testid="stPopover"]' in helper_source
    assert "row-gap: 4px" in helper_source
    assert "gap: 4px" in helper_source


def test_date_picker_horizontal_block_alignment_css_is_scoped() -> None:
    """날짜 선택기가 있는 가로 블록만 컬럼 요소 위치를 일괄 정렬하는 CSS를 갖는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert 'div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"])' in helper_source
    assert 'div[data-testid="column"]' in helper_source
    assert "justify-content: flex-end" in helper_source
    assert "align-self: stretch" in helper_source


def test_weekly_pages_remove_duplicated_end_date_control() -> None:
    """주간 페이지가 주 선택 picker 외 별도 종료일 UI를 표시하지 않는지 검증한다."""
    page_paths = [
        Path("dev_pages/07_weekly_analysis.py"),
        Path("dev_pages/08_weekly_interpretation.py"),
        Path("dev_pages/09_weekly_feedback.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert "select_week_range" in page_source
        assert "render_readonly_control" not in page_source
        assert "분석 주 종료일" not in page_source


def test_weekly_range_picker_uses_week_selection_and_sunday_range() -> None:
    """주간 선택 헬퍼가 week picker를 유지하고 UI와 같은 일~토 범위를 계산하는지 검증한다."""
    from catcher_llm.ui import date_picker as date_picker_module

    helper_source = Path(date_picker_module.__file__).read_text(encoding="utf-8")

    assert "def select_week_range" in helper_source
    select_week_range_source = helper_source.split("def select_week_range", maxsplit=1)[1].split(
        "def select_month",
        maxsplit=1,
    )[0]
    assert "picker_type=PickerType.week" in select_week_range_source
    assert "picker_type=PickerType.date" not in select_week_range_source


def test_date_picker_bundle_height_patch_source() -> None:
    """streamlit-date-picker 번들의 높이 하드코딩을 400px로 바꾸는지 검증한다."""
    from catcher_llm.ui.date_picker import _patch_date_picker_bundle_source

    source = 'height:"50px";height:"200px";height:"350px";height:"360px";height:"520px"'

    patched_source, changed = _patch_date_picker_bundle_source(source)

    assert changed is True
    assert 'height:"50px"' not in patched_source
    assert 'height:"200px"' not in patched_source
    assert 'height:"350px"' not in patched_source
    assert 'height:"360px"' not in patched_source
    assert 'height:"520px"' not in patched_source
    assert patched_source.count('height:"400px"') == 5


def test_daily_analysis_dev_page_renders_payment_behavior_metrics() -> None:
    """일일 분석 개발 페이지가 지출 마찰력과 결제 밀도 지표를 표시하는지 검증한다."""
    with patch(
        "catcher_llm.services.consumption_feedback.daily_analysis."
        "build_daily_consumption_analysis_json",
        return_value=_make_daily_analysis_result(),
    ):
        app = AppTest.from_file("dev_pages/04_daily_analysis.py")
        app.run(timeout=10)
        _click_button_by_label(app, "일일 분석 실행")

    assert len(app.exception) == 0
    assert any(subheader.value == "지출 마찰력 및 결제 밀도" for subheader in app.subheader)
    assert any(subheader.value == "문서 기준 일일 추가 지표" for subheader in app.subheader)


def test_consumption_analysis_pages_render_period_document_metrics() -> None:
    """일일·주간·월간 분석 페이지가 중복 없는 문서 기준 추가 지표만 렌더링하는지 검증한다."""
    page_expectations = {
        Path("dev_pages/04_daily_analysis.py"): {
            "included": [
                "문서 기준 일일 추가 지표",
                "daily_metrics",
                "일일 예산 소진율",
                "충동소비 점수",
            ],
            "excluded": ["일일 총 소비금액", "일일 거래 건수", "일일 평균 거래금액"],
        },
        Path("dev_pages/07_weekly_analysis.py"): {
            "included": [
                "문서 기준 주간 추가 지표",
                "weekly_metrics",
                "주간 소비 변동성",
                "주말 과소비 지수",
            ],
            "excluded": ["주간 총 소비금액", "전주 대비 소비 증감률", "문서 기준 요일별 소비 패턴"],
        },
        Path("dev_pages/10_monthly_analysis.py"): {
            "included": [
                "문서 기준 월간 추가 지표",
                "monthly_metrics",
                "고정비 부담률",
                "구독료 합계",
            ],
            "excluded": ["월간 총 소비금액", "전월 대비 소비 증감률", "카테고리별 월간 소비 비중"],
        },
    }

    for page_path, expected_fragments in page_expectations.items():
        page_source = page_path.read_text(encoding="utf-8")
        for expected_fragment in expected_fragments["included"]:
            assert expected_fragment in page_source
        for duplicated_fragment in expected_fragments["excluded"]:
            assert duplicated_fragment not in page_source


def test_consumption_analysis_pages_render_extended_comparisons() -> None:
    """일일·주간·월간 분석 페이지가 새 기간 비교 블록을 화면에 노출하는지 검증한다."""
    page_expectations = {
        Path("dev_pages/04_daily_analysis.py"): [
            "확장 일일 비교",
            "daily_comparisons",
            "지난주 같은 요일 대비",
            "최근 4주 같은 요일 평균 대비",
            "reference_days",
        ],
        Path("dev_pages/07_weekly_analysis.py"): [
            "확장 주간 비교",
            "weekly_comparisons",
            "최근 4주 평균 대비",
            "지난달 같은 주차 대비",
            "reference_periods",
        ],
        Path("dev_pages/10_monthly_analysis.py"): [
            "확장 월간 비교",
            "monthly_comparisons",
            "최근 3개월 평균 대비",
            "reference_month_details",
        ],
    }

    for page_path, expected_fragments in page_expectations.items():
        page_source = page_path.read_text(encoding="utf-8")
        for expected_fragment in expected_fragments:
            assert expected_fragment in page_source


def test_consumption_analysis_pages_render_user_financial_metrics() -> None:
    """일일·주간·월간 분석 페이지가 사용자 연봉·목표 소비 기반 지표를 노출하는지 검증한다."""
    page_expectations = {
        Path("dev_pages/04_daily_analysis.py"): [
            "일일 잔여 예산",
            "일일 예산 초과액",
            "일 환산 소득 대비 소비율",
            "월 누적 목표 사용률",
            "월말 예상 소비",
            "월말 예상 목표 사용률",
            "월말까지 하루 허용 소비",
            "daily_remaining_budget",
            "required_daily_budget_until_month_end",
        ],
        Path("dev_pages/07_weekly_analysis.py"): [
            "주간 잔여 예산",
            "주간 예산 초과액",
            "주 환산 소득 대비 소비율",
            "주간 예산 소진 배율",
            "월 누적 목표 사용률",
            "주간 페이스 기준 월말 예상 소비",
            "weekly_remaining_budget",
            "projected_monthly_spending_from_weekly_pace",
        ],
        Path("dev_pages/10_monthly_analysis.py"): [
            "월간 잔여 예산",
            "월간 예산 초과액",
            "월소득 대비 총소비율",
            "목표 소비 한도 소득 비중",
            "추정 저축액",
            "추정 저축률",
            "목표 달성 시 저축액",
            "목표 달성 시 저축률",
            "소비 여력",
            "비필수 소비 소득 비중",
            "monthly_remaining_budget",
            "nonessential_spending_income_rate_percent",
        ],
    }

    for page_path, expected_fragments in page_expectations.items():
        page_source = page_path.read_text(encoding="utf-8")
        for expected_fragment in expected_fragments:
            assert expected_fragment in page_source


def test_consumption_interpretation_pages_render_user_financial_metrics() -> None:
    """일일·주간·월간 해석 페이지가 체인 입력에 들어가는 소득·목표 소비 지표를 노출하는지 검증한다."""
    page_expectations = {
        Path("dev_pages/05_daily_interpretation.py"): [
            "소득·목표 소비 지표",
            "일일 잔여 예산",
            "일일 예산 초과액",
            "일 환산 소득 대비 소비율",
            "월 누적 목표 사용률",
            "월말 예상 소비",
            "월말 예상 목표 사용률",
            "월말까지 남은 하루 허용 소비",
        ],
        Path("dev_pages/08_weekly_interpretation.py"): [
            "소득·목표 소비 지표",
            "주간 잔여 예산",
            "주간 예산 초과액",
            "주 환산 소득 대비 소비율",
            "주간 예산 소진 배율",
            "월 누적 목표 사용률",
            "주간 페이스 기준 월말 예상 소비",
        ],
        Path("dev_pages/11_monthly_interpretation.py"): [
            "소득·목표 소비 지표",
            "월간 잔여 예산",
            "월간 예산 초과액",
            "월소득 대비 총소비율",
            "목표 소비 한도 소득 비중",
            "추정 저축액",
            "추정 저축률",
            "목표 달성 시 저축액",
            "목표 달성 시 저축률",
            "소비 여력",
            "비필수 소비 소득 비중",
        ],
    }

    for page_path, expected_fragments in page_expectations.items():
        page_source = page_path.read_text(encoding="utf-8")
        for expected_fragment in expected_fragments:
            assert expected_fragment in page_source


def test_consumption_interpretation_dev_page_renders_payment_behavior_metrics() -> None:
    """소비 해석 개발 페이지가 추출 지표 요약에 새 결제 행동 지표를 표시하는지 검증한다."""
    app = AppTest.from_file("dev_pages/05_daily_interpretation.py")

    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(subheader.value == "지출 마찰력 및 결제 밀도" for subheader in app.subheader)


def test_daily_interpretation_page_uses_default_sqlite_input_without_sample_selector() -> None:
    """일일 해석 페이지가 샘플 JSON 선택 없이 기본 SQLite 분석 입력을 사용하는지 검증한다."""
    page_source = Path("dev_pages/05_daily_interpretation.py").read_text(encoding="utf-8")

    assert "SAMPLE_JSON_PATH" not in page_source
    assert "load_user_spending_data" not in page_source
    assert "st.radio(" not in page_source
    assert "노트북 샘플 JSON" not in page_source
    assert "입력 데이터" not in page_source
    assert "전일 비교 기준일" not in page_source
    assert "previous_day = analysis_day - timedelta(days=1)" in page_source


def test_daily_feedback_dev_page_renders_profile_and_memory_context() -> None:
    """일일 피드백 개발 페이지가 서비스의 사용자 프로필과 메모리 컨텍스트를 표시하는지 검증한다."""
    fake_result = DailyFeedbackServiceResult(
        member_id=1,
        analysis_date="2024-04-01",
        feedback=DailyFeedbackResult(
            summary_title="오늘은 생활비를 확인하세요",
            scolding_message="생활 카테고리 소비가 커졌습니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="내일 오전 장보기 목록을 확인합니다.",
        ),
        user_profile=UserProfileContext(
            user_id=1,
            name="김토스",
            job="개발자",
            saving_goal_text="비상금 300만원 만들기",
        ),
        memory_context=DailyFeedbackMemoryContext(
            user_id=1,
            memory_summary="최근 식비와 쇼핑 지출이 반복적으로 높다.",
            recent_sessions=[
                DailyFeedbackSessionContext(
                    analysis_date="2024-03-31",
                    todo_tomorrow="간식 결제를 줄인다.",
                )
            ],
        ),
        retrieval_queries=["비상금 300만원 만들기 목표 소비 절약 방법"],
        daily_analysis=UserSpendingData.model_validate(_make_daily_analysis_result()),
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_daily_session_for_date",
            return_value=None,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
            return_value=fake_result,
        ),
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "일일 피드백 생성")

    assert len(app.exception) == 0
    assert any(subheader.value == "지출 마찰력 및 결제 밀도" for subheader in app.subheader)
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)
    assert any(expander.label == "메모리/세션 컨텍스트 JSON" for expander in app.expander)


def test_daily_feedback_dev_page_can_regenerate_cached_session() -> None:
    """저장된 일일 세션이 있어도 재생성 버튼이 새 피드백 생성을 호출하는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-04-01",
        period_type="daily",
        feedback_message="기존 저장 피드백입니다.",
        feedback_reason="[]",
        todo_tomorrow="기존 미션입니다.",
    )
    fake_result = DailyFeedbackServiceResult(
        member_id=1,
        analysis_date="2026-04-01",
        feedback=DailyFeedbackResult(
            summary_title="재생성된 피드백",
            scolding_message="새로 생성한 피드백입니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="새 미션입니다.",
        ),
        user_profile=UserProfileContext(user_id=1, name="김토스"),
        memory_context=DailyFeedbackMemoryContext(user_id=1),
        retrieval_queries=["재생성 테스트 질의"],
        daily_analysis=UserSpendingData.model_validate(_make_daily_analysis_result()),
    )
    call_kwargs: dict[str, object] = {}

    def fake_generate_daily_feedback(*args: object, **kwargs: object) -> DailyFeedbackServiceResult:
        """재생성 버튼 클릭 시 전달된 피드백 생성 인자를 저장하고 가짜 결과를 반환한다."""
        call_kwargs.update(kwargs)
        return fake_result

    with (
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_daily_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_all_daily_sessions",
            return_value=[cached_session],
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
            side_effect=fake_generate_daily_feedback,
        ),
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "일일 피드백 재생성")

    assert len(app.exception) == 0
    assert call_kwargs["analysis_date"].isoformat() == "2026-04-01"
    assert any(subheader.value == "재생성된 피드백" for subheader in app.subheader)
    assert any(expander.label == "최종 피드백 JSON" for expander in app.expander)


def test_daily_feedback_dev_page_loads_partial_cached_session() -> None:
    """피드백 본문이 비어 있어도 저장된 일일 산출물이 있으면 캐시 세션으로 불러오는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-04-01",
        period_type="daily",
        analysis_result='{"stable_metrics": {"today_total": 1000}}',
        feedback_message="",
        feedback_reason="[]",
        todo_tomorrow="저장된 미션입니다.",
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_daily_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_all_daily_sessions",
            return_value=[cached_session],
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
        ) as generate_feedback,
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)

    assert len(app.exception) == 0
    generate_feedback.assert_not_called()
    assert any(button.label == "일일 피드백 재생성" for button in app.button)
    assert not any(button.label == "일일 피드백 생성" for button in app.button)


def test_weekly_feedback_dev_page_can_regenerate_cached_session() -> None:
    """저장된 주간 세션이 있어도 재생성 버튼이 새 피드백 생성을 호출하는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-03-29",
        period_type="weekly",
        feedback_message="기존 저장 주간 피드백입니다.",
        feedback_reason="[]",
        todo_tomorrow="기존 다음 주 미션입니다.",
    )
    fake_result = WeeklyFeedbackServiceResult(
        member_id=1,
        week_start="2026-03-29",
        week_end="2026-04-04",
        feedback=WeeklyFeedbackResult(
            summary_title="재생성된 주간 피드백",
            feedback_message="새로 생성한 주간 피드백입니다.",
            key_evidences=[],
            action_items=[],
            next_week_mission="새 다음 주 미션입니다.",
        ),
        user_profile=UserProfileContext(user_id=1, name="김토스"),
        retrieval_queries=["주간 재생성 테스트 질의"],
    )
    call_kwargs: dict[str, object] = {}

    def fake_generate_weekly_feedback(
        *args: object, **kwargs: object
    ) -> WeeklyFeedbackServiceResult:
        """재생성 버튼 클릭 시 전달된 주간 피드백 생성 인자를 저장하고 가짜 결과를 반환한다."""
        call_kwargs.update(kwargs)
        return fake_result

    with (
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.load_weekly_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.generate_weekly_feedback",
            side_effect=fake_generate_weekly_feedback,
        ),
    ):
        app = AppTest.from_file("dev_pages/09_weekly_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "주간 피드백 재생성")

    assert len(app.exception) == 0
    assert call_kwargs["week_start"].isoformat() == "2026-03-29"
    assert call_kwargs["week_end"].isoformat() == "2026-04-04"
    assert any(subheader.value == "재생성된 주간 피드백" for subheader in app.subheader)
    assert any(expander.label == "최종 피드백 JSON" for expander in app.expander)


def test_weekly_feedback_dev_page_loads_partial_cached_session() -> None:
    """피드백 본문이 비어 있어도 저장된 주간 산출물이 있으면 캐시 세션으로 불러오는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-03-29",
        period_type="weekly",
        analysis_result='{"weekly_summary": {"this_week_total": 1000}}',
        feedback_message="",
        feedback_reason="[]",
        todo_tomorrow="저장된 다음 주 미션입니다.",
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.load_weekly_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.generate_weekly_feedback",
        ) as generate_feedback,
    ):
        app = AppTest.from_file("dev_pages/09_weekly_feedback.py")
        app.run(timeout=10)

    assert len(app.exception) == 0
    generate_feedback.assert_not_called()
    assert any(button.label == "주간 피드백 재생성" for button in app.button)
    assert not any(button.label == "주간 피드백 생성" for button in app.button)


def test_monthly_feedback_dev_page_can_regenerate_cached_session() -> None:
    """저장된 월간 세션이 있어도 재생성 버튼이 새 피드백 생성을 호출하는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-04",
        period_type="monthly",
        feedback_message="기존 저장 월간 피드백입니다.",
        feedback_reason="[]",
        todo_tomorrow="기존 다음 달 미션입니다.",
    )
    fake_result = MonthlyFeedbackServiceResult(
        member_id=1,
        analysis_month="2026-04",
        feedback=MonthlyFeedbackResult(
            summary_title="재생성된 월간 피드백",
            feedback_message="새로 생성한 월간 피드백입니다.",
            key_evidences=[],
            action_items=[],
            next_month_mission="새 다음 달 미션입니다.",
        ),
        user_profile=UserProfileContext(user_id=1, name="김토스"),
        retrieval_queries=["월간 재생성 테스트 질의"],
    )
    call_kwargs: dict[str, object] = {}

    def fake_generate_monthly_feedback(
        *args: object, **kwargs: object
    ) -> MonthlyFeedbackServiceResult:
        """재생성 버튼 클릭 시 전달된 월간 피드백 생성 인자를 저장하고 가짜 결과를 반환한다."""
        call_kwargs.update(kwargs)
        return fake_result

    with (
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.load_monthly_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.generate_monthly_feedback",
            side_effect=fake_generate_monthly_feedback,
        ),
    ):
        app = AppTest.from_file("dev_pages/12_monthly_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "월간 피드백 재생성")

    assert len(app.exception) == 0
    assert call_kwargs["analysis_month"] == "2026-04"
    assert any(subheader.value == "재생성된 월간 피드백" for subheader in app.subheader)
    assert any(expander.label == "최종 피드백 JSON" for expander in app.expander)


def test_monthly_feedback_dev_page_loads_partial_cached_session() -> None:
    """피드백 본문이 비어 있어도 저장된 월간 산출물이 있으면 캐시 세션으로 불러오는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-04",
        period_type="monthly",
        analysis_result='{"monthly_summary": {"this_month_total": 1000}}',
        feedback_message="",
        feedback_reason="[]",
        todo_tomorrow="저장된 다음 달 미션입니다.",
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.load_monthly_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.generate_monthly_feedback",
        ) as generate_feedback,
    ):
        app = AppTest.from_file("dev_pages/12_monthly_feedback.py")
        app.run(timeout=10)

    assert len(app.exception) == 0
    generate_feedback.assert_not_called()
    assert any(button.label == "월간 피드백 재생성" for button in app.button)
    assert not any(button.label == "월간 피드백 생성" for button in app.button)


def test_feedback_dev_pages_render_feedback_reaction_controls() -> None:
    """일·주·월 피드백 개발 페이지가 피드백 반응 저장 UI를 호출하는지 검증한다."""
    page_period_snippets = {
        "dev_pages/06_daily_feedback.py": 'period_type="daily"',
        "dev_pages/09_weekly_feedback.py": 'period_type="weekly"',
        "dev_pages/12_monthly_feedback.py": 'period_type="monthly"',
    }

    for page_path, period_snippet in page_period_snippets.items():
        page_source = Path(page_path).read_text(encoding="utf-8")
        assert "render_feedback_reaction_controls" in page_source
        assert period_snippet in page_source


def test_feedback_dev_pages_render_generation_progress() -> None:
    """일·주·월 피드백/리포트 페이지가 생성 단계 진행 바와 타이밍 콜백을 연결하는지 검증한다."""
    page_paths = [
        Path("dev_pages/06_daily_feedback.py"),
        Path("dev_pages/09_weekly_feedback.py"),
        Path("dev_pages/12_monthly_feedback.py"),
        Path("pages/04_daily_report.py"),
        Path("pages/05_weekly_report.py"),
        Path("pages/06_monthly_report.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert "create_feedback_progress_callback" in page_source
        assert "timing_callback=progress_callback" in page_source


def test_report_pages_render_generated_feedback_fields() -> None:
    """리포트 페이지가 피드백 개발 페이지와 같은 생성 피드백 필드를 메인 화면에 반영하는지 검증한다."""
    daily_source = Path("pages/04_daily_report.py").read_text(encoding="utf-8")
    weekly_source = Path("pages/05_weekly_report.py").read_text(encoding="utf-8")
    monthly_source = Path("pages/06_monthly_report.py").read_text(encoding="utf-8")

    assert "feedback.scolding_message" in daily_source
    assert "feedback.tomorrow_mission" in daily_source
    assert "feedback.key_evidences" in daily_source
    assert "feedback.action_items" in daily_source

    assert "feedback.feedback_message" in weekly_source
    assert "feedback.next_week_mission" in weekly_source
    assert "feedback.key_evidences" in weekly_source
    assert "feedback.action_items" in weekly_source
    assert 'getattr(feedback, "scolding_message"' not in weekly_source

    assert "feedback.feedback_message" in monthly_source
    assert "feedback.next_month_mission" in monthly_source
    assert "feedback.key_evidences" in monthly_source
    assert "feedback.action_items" in monthly_source
    assert 'getattr(feedback, "scolding_message"' not in monthly_source


def test_streamlit_pages_do_not_use_deprecated_container_width_argument() -> None:
    """Streamlit 페이지 코드에서 제거 예정인 use_container_width 인자를 쓰지 않는지 검증한다."""
    page_paths = [
        Path("app.py"),
        *Path("pages").glob("*.py"),
        *Path("dev_pages").glob("*.py"),
    ]

    offenders = [
        str(page_path)
        for page_path in page_paths
        if "use_container_width" in page_path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_main_profile_page_renders_long_persona_as_collapsible_panel() -> None:
    """메인 사용자 프로필 페이지가 긴 페르소나를 접이식 패널로 표시하는지 검증한다."""
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert 'st.metric("최상위 카드 등급"' not in app_source
    assert 'st.metric("페르소나"' not in app_source
    assert 'persona_text = profile.get("persona")' in app_source
    assert "나의 페르소나" in app_source
    assert "아직 등록된 페르소나가 없습니다." in app_source
    assert "with st.expander(title, expanded=expanded):" in app_source
    assert "collapsible=True" in app_source


def test_signup_page_uses_korean_labels_and_grouped_layout() -> None:
    """회원가입 페이지가 사용자 입력 라벨을 한국어로 표시하고 섹션별로 배치하는지 검증한다."""
    page_source = Path("pages/00_user_signup.py").read_text(encoding="utf-8")

    expected_labels = [
        '"이름"',
        '"나이"',
        '"성별"',
        '"직업"',
        '"연소득"',
        '"거주 지역"',
        '"월 목표 최대 소비 금액"',
        '"페르소나"',
        '"절약 목표"',
    ]
    for label in expected_labels:
        assert label in page_source

    legacy_label_snippets = [
        'st.text_input("name"',
        'st.number_input("age"',
        'st.selectbox("gender"',
        'st.text_input("occupation"',
        'st.number_input("annual_income"',
        'st.text_input("region"',
        'st.number_input("target_max_spending_amount"',
        'st.text_area("persona"',
        'st.text_area("saving_goal_text"',
    ]
    for label in legacy_label_snippets:
        assert label not in page_source

    assert 'st.markdown("#### 기본 정보")' in page_source
    assert 'st.markdown("#### 소득과 목표")' in page_source
    assert 'st.markdown("#### 상세 프로필")' in page_source


def test_csv_upload_page_saves_uploaded_transactions_to_sqlite() -> None:
    """CSV 업로드 페이지가 업로드 데이터를 transactions 테이블 저장 서비스로 전달하는지 검증한다."""
    page_source = Path("pages/01_csv_upload.py").read_text(encoding="utf-8")

    assert "upload_transactions_dataframe" in page_source
    assert "감지된 컬럼 매핑" in page_source
    assert "거래 내역 DB 저장" in page_source
    assert "st.session_state.user_id" in page_source


def test_daily_feedback_reaction_button_opens_reason_input_and_saves_reaction() -> None:
    """일일 피드백 반응 버튼을 누르면 반응 저장 후 사유 입력 UI가 열리는지 검증한다."""
    cached_session = SessionModel(
        user_id=1,
        analysis_date="2026-04-01",
        period_type="daily",
        feedback_message="기존 저장 피드백입니다.",
        feedback_reason="[]",
        todo_tomorrow="기존 미션입니다.",
    )
    reaction_calls: list[dict[str, object]] = []

    def fake_save_session_feedback_reaction(
        *args: object, **kwargs: object
    ) -> FeedbackReactionSaveResult:
        """피드백 반응 저장 호출 인자를 기록하고 저장 결과를 반환한다."""
        reaction_calls.append(kwargs)
        return FeedbackReactionSaveResult(
            member_id=1,
            analysis_date="2026-04-01",
            period_type="daily",
            reaction="like",
            reason=None,
        )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_daily_session_for_date",
            return_value=cached_session,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_all_daily_sessions",
            return_value=[cached_session],
        ),
        patch(
            "catcher_llm.ui.feedback_reaction.save_session_feedback_reaction",
            side_effect=fake_save_session_feedback_reaction,
        ),
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "좋아요")

    assert len(app.exception) == 0
    assert reaction_calls[0]["reaction"] == "like"
    assert reaction_calls[0]["reason"] == ""
    assert any(text_area.label == "반응 이유" for text_area in app.text_area)


def test_daily_feedback_timing_dev_page_renders_step_records() -> None:
    """일일 피드백 소요 시간 페이지가 타이밍 콜백 기록을 화면에 표시하는지 검증한다."""
    fake_result = DailyFeedbackServiceResult(
        member_id=1,
        analysis_date="2024-04-01",
        feedback=DailyFeedbackResult(
            summary_title="오늘은 생활비를 확인하세요",
            scolding_message="생활 카테고리 소비가 커졌습니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="내일 오전 장보기 목록을 확인합니다.",
        ),
    )

    def fake_generate_daily_feedback(*args: object, **kwargs: object) -> DailyFeedbackServiceResult:
        """테스트용 타이밍 기록을 콜백으로 전달하고 가짜 피드백 결과를 반환한다."""
        timing_callback = kwargs.get("timing_callback")
        if callable(timing_callback):
            timing_callback(
                DailyFeedbackTimingRecord(
                    step_key="daily_analysis",
                    step_name="일일 소비 분석 JSON 생성",
                    elapsed_seconds=0.12,
                    detail="SQLite transactions 조회와 pandas 지표 계산",
                )
            )
            timing_callback(
                DailyFeedbackTimingRecord(
                    step_key="feedback_chain",
                    step_name="최종 피드백 체인 실행",
                    elapsed_seconds=1.34,
                    detail="분석/해석/RAG/프로필/메모리 기반 구조화 LLM 호출",
                )
            )
        return fake_result

    with patch(
        "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
        side_effect=fake_generate_daily_feedback,
    ):
        app = AppTest.from_file("dev_pages/17_daily_feedback_timing.py")
        app.run(timeout=10)
        _click_button_by_label(app, "타이밍 측정 실행")

    assert len(app.exception) == 0
    assert any(subheader.value == "단계별 소요 시간" for subheader in app.subheader)
    assert any(metric.label == "총 소요 시간" for metric in app.metric)
    assert any(expander.label == "최종 피드백 JSON" for expander in app.expander)


def test_daily_interpretation_compare_dev_page_renders_mode_results() -> None:
    """일일 해석 방식 비교 페이지가 세 모드의 실행 결과를 표시하는지 검증한다."""
    fake_user_data = UserSpendingData.model_validate(_make_daily_analysis_result())
    fake_profile = UserProfileContext(
        user_id=1,
        name="김토스",
        saving_goal_text="비상금 300만원 만들기",
    )

    class FakeInterpretationChain:
        """테스트용 해석 체인 결과를 고정 dict로 반환한다."""

        def __init__(self, mode: str) -> None:
            """해석 모드 이름을 결과에 포함할 수 있게 저장한다."""
            self.mode = mode

        def invoke(self, payload: dict[str, str]) -> dict[str, object]:
            """해석 입력을 받아 모드 식별 가능한 결과 dict를 반환한다."""
            return {
                "pattern_result": {"mode": self.mode, "payload_keys": sorted(payload)},
                "problem_result": {"mode": self.mode},
                "cause_result": {"mode": self.mode},
                "action_result": {"mode": self.mode},
            }

    with (
        patch(
            "catcher_llm.services.consumption_feedback.daily_analysis."
            "build_daily_consumption_analysis_json",
            return_value=_make_daily_analysis_result(),
        ),
        patch(
            "catcher_llm.services.consumption_feedback.interpretation.parse_user_spending_data",
            return_value=fake_user_data,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.daily_feedback.load_user_profile_context",
            return_value=fake_profile,
        ),
        patch(
            "catcher_llm.chains.consumption_feedback.build_spending_analysis_chain",
            return_value=FakeInterpretationChain("split"),
        ),
        patch(
            "catcher_llm.chains.consumption_feedback.build_balanced_spending_analysis_chain",
            return_value=FakeInterpretationChain("balanced"),
        ),
        patch(
            "catcher_llm.chains.consumption_feedback.build_unified_spending_analysis_chain",
            return_value=FakeInterpretationChain("unified"),
        ),
    ):
        app = AppTest.from_file("dev_pages/18_daily_interpretation_compare.py")
        app.run(timeout=10)
        _click_button_by_label(app, "해석 방식 비교 실행")

    assert len(app.exception) == 0
    assert any(subheader.value == "모드별 실행 시간" for subheader in app.subheader)
    assert any(expander.label == "기존 방식 결과 JSON" for expander in app.expander)
    assert any(expander.label == "균형 방식 결과 JSON" for expander in app.expander)
    assert any(expander.label == "통합 방식 결과 JSON" for expander in app.expander)


def test_daily_feedback_unified_dev_page_uses_unified_interpretation_mode() -> None:
    """일일 통합 피드백 페이지가 최종 피드백 생성 시 통합 해석 모드를 사용하는지 검증한다."""
    fake_result = DailyFeedbackServiceResult(
        member_id=1,
        analysis_date="2026-04-01",
        feedback=DailyFeedbackResult(
            summary_title="통합 해석 피드백",
            scolding_message="통합 해석 체인으로 만든 피드백입니다.",
            key_evidences=[],
            action_items=[],
            tomorrow_mission="내일 오전 소비 계획을 확인합니다.",
        ),
        daily_analysis=UserSpendingData.model_validate(_make_daily_analysis_result()),
        retrieval_queries=["생활 소비 절약 방법"],
    )
    call_kwargs: dict[str, object] = {}

    def fake_generate_daily_feedback(*args: object, **kwargs: object) -> DailyFeedbackServiceResult:
        """통합 모드 호출 인자를 저장하고 가짜 피드백 결과를 반환한다."""
        call_kwargs.update(kwargs)
        timing_callback = kwargs.get("timing_callback")
        if callable(timing_callback):
            timing_callback(
                DailyFeedbackTimingRecord(
                    step_key="interpretation_chain",
                    step_name="소비 해석 체인 실행 (unified)",
                    elapsed_seconds=0.42,
                    detail="통합 해석 체인 테스트",
                )
            )
        return fake_result

    with patch(
        "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
        side_effect=fake_generate_daily_feedback,
    ):
        app = AppTest.from_file("dev_pages/19_daily_feedback_unified.py")
        app.run(timeout=10)
        _click_button_by_label(app, "통합 방식 일일 피드백 생성")

    assert len(app.exception) == 0
    assert call_kwargs["interpretation_mode"] == "unified"
    assert any(subheader.value == "통합 방식 피드백 결과" for subheader in app.subheader)
    assert any(expander.label == "최종 피드백 JSON" for expander in app.expander)


def test_weekly_interpretation_dev_page_renders_weekly_indicators() -> None:
    """주간 소비 해석 개발 페이지가 주간 지표 요약과 체인 입력 JSON을 표시하는지 검증한다."""
    app = AppTest.from_file("dev_pages/08_weekly_interpretation.py")

    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(subheader.value == "주간 핵심 소비 지표" for subheader in app.subheader)
    assert any(subheader.value == "주간 카테고리 증감" for subheader in app.subheader)
    assert any(expander.label == "체인 입력 JSON" for expander in app.expander)


def test_weekly_feedback_dev_page_renders_feedback_and_contexts() -> None:
    """주간 피드백 개발 페이지가 최종 피드백과 RAG 컨텍스트를 표시하는지 검증한다."""
    fake_result = WeeklyFeedbackServiceResult(
        member_id=1,
        week_start="2024-04-01",
        week_end="2024-04-07",
        feedback=WeeklyFeedbackResult(
            summary_title="이번 주는 식비를 줄이세요",
            feedback_message="배달과 카페 지출이 반복됐습니다.",
            key_evidences=[],
            action_items=[],
            next_week_mission="다음 주 배달 주문은 1회만 허용합니다.",
        ),
        user_profile=UserProfileContext(
            user_id=1,
            name="김토스",
            saving_goal_text="비상금 300만원 만들기",
        ),
        retrieval_queries=["배달 소비 절약 방법"],
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.load_weekly_session_for_date",
            return_value=None,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.weekly_feedback.generate_weekly_feedback",
            return_value=fake_result,
        ),
    ):
        app = AppTest.from_file("dev_pages/09_weekly_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "주간 피드백 생성")

    assert len(app.exception) == 0
    assert any(subheader.value == "피드백 근거" for subheader in app.subheader)
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)


def test_monthly_interpretation_dev_page_renders_monthly_indicators() -> None:
    """월간 소비 해석 개발 페이지가 월간 지표 요약과 체인 입력 JSON을 표시하는지 검증한다."""
    app = AppTest.from_file("dev_pages/11_monthly_interpretation.py")

    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(subheader.value == "월간 핵심 소비 지표" for subheader in app.subheader)
    assert any(subheader.value == "월간 카테고리 증감" for subheader in app.subheader)
    assert any(expander.label == "체인 입력 JSON" for expander in app.expander)


def test_monthly_feedback_dev_page_renders_feedback_and_contexts() -> None:
    """월간 피드백 개발 페이지가 최종 피드백과 RAG 컨텍스트를 표시하는지 검증한다."""
    fake_result = MonthlyFeedbackServiceResult(
        member_id=1,
        analysis_month="2024-04",
        feedback=MonthlyFeedbackResult(
            summary_title="이번 달은 고정비를 확인하세요",
            feedback_message="자동이체와 배달 소비가 월간 지출을 키웠습니다.",
            key_evidences=[],
            action_items=[],
            next_month_mission="다음 달 첫날 자동이체 목록을 정리합니다.",
        ),
        user_profile=UserProfileContext(
            user_id=1,
            name="김토스",
            saving_goal_text="비상금 300만원 만들기",
        ),
        retrieval_queries=["고정비 절약 방법"],
    )

    with (
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.load_monthly_session_for_date",
            return_value=None,
        ),
        patch(
            "catcher_llm.services.consumption_feedback.monthly_feedback.generate_monthly_feedback",
            return_value=fake_result,
        ),
    ):
        app = AppTest.from_file("dev_pages/12_monthly_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "월간 피드백 생성")

    assert len(app.exception) == 0
    assert any(subheader.value == "피드백 근거" for subheader in app.subheader)
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)
