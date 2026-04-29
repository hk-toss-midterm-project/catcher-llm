from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

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
        "일간 보고서",
        "주간 보고서",
        "월간 보고서",
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
    ]
    assert all(spec.path.is_file() for spec in specs)


def test_dev_app_renders_default_page_without_exception() -> None:
    """Streamlit 개발 앱의 기본 페이지가 import 예외 없이 렌더링되는지 검증한다."""
    app = AppTest.from_file("dev_app.py")

    app.run(timeout=10)

    assert len(app.exception) == 0


def test_daily_report_page_uses_sqlite_backed_default_selection() -> None:
    """일간 보고서 페이지가 하드코딩 날짜 대신 SQLite 거래 기반 기본값을 쓰는지 검증한다."""
    page_source = Path("dev_pages/13_daily_report_rim.py").read_text(encoding="utf-8")

    assert "get_default_daily_report_selection" in page_source
    assert 'value="1"' not in page_source
    assert "date(2024, 3, 31)" not in page_source
    assert "st.number_input(" in page_source
    assert '"Member ID"' in page_source


def test_daily_report_page_uses_configured_sqlite_for_points() -> None:
    """일간 보고서 포인트 조회와 갱신이 설정 SQLite 경로와 v3 점수 컬럼을 쓰는지 검증한다."""
    page_source = Path("dev_pages/13_daily_report_rim.py").read_text(encoding="utf-8")

    assert "get_settings().sqlite_db_path" in page_source
    assert "personal_score" in page_source
    assert r"C:\Users\user\catcher" not in page_source
    assert '"개인 점수"' not in page_source


def test_consumption_dev_pages_use_2026_january_calendar_defaults() -> None:
    """소비 개발 페이지의 달력 기본 월이 v3 데이터 시작 월인 2026년 1월인지 검증한다."""
    from datetime import date

    from catcher_llm.ui.date_picker import DEFAULT_CALENDAR_DATE, DEFAULT_CALENDAR_MONTH

    assert DEFAULT_CALENDAR_DATE == date(2026, 1, 1)
    assert DEFAULT_CALENDAR_MONTH == "2026-01"

    daily_or_weekly_pages = [
        Path("dev_pages/04_daily_analysis.py"),
        Path("dev_pages/05_daily_interpretation.py"),
        Path("dev_pages/06_daily_feedback.py"),
        Path("dev_pages/07_weekly_analysis.py"),
        Path("dev_pages/08_weekly_interpretation.py"),
        Path("dev_pages/09_weekly_feedback.py"),
        Path("dev_pages/14_weekly_report_rim.py"),
    ]
    for page_path in daily_or_weekly_pages:
        page_source = page_path.read_text(encoding="utf-8")
        assert "DEFAULT_CALENDAR_DATE" in page_source
        assert "date(2024" not in page_source

    monthly_pages = [
        Path("dev_pages/10_monthly_analysis.py"),
        Path("dev_pages/11_monthly_interpretation.py"),
        Path("dev_pages/12_monthly_feedback.py"),
        Path("dev_pages/15_monthly_report_rim.py"),
    ]
    for page_path in monthly_pages:
        page_source = page_path.read_text(encoding="utf-8")
        assert "DEFAULT_CALENDAR_MONTH" in page_source
        assert '"2024-04"' not in page_source
        assert '"2024-03"' not in page_source


def test_consumption_dev_pages_use_popover_date_picker_helper() -> None:
    """일일·주간·월간 소비 개발 페이지가 popover 날짜 선택 헬퍼로 날짜를 선택하는지 검증한다."""
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
        date(2024, 4, 1),
        date(2024, 4, 7),
    )
    assert _coerce_week_range("2024-14th", default) == (
        date(2024, 4, 1),
        date(2024, 4, 7),
    )
    assert _coerce_month("2024-04-18", default) == "2024-04"
    assert _coerce_month("2024-04", default) == "2024-04"
    assert _coerce_picker_date(PickerType.week, "2024-14th", default) == date(2024, 4, 1)
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


def test_weekly_controls_use_readonly_box_for_end_date() -> None:
    """주간 컨트롤 행의 종료일 표시가 metric 대신 공통 읽기 전용 박스를 쓰는지 검증한다."""
    page_paths = [
        Path("dev_pages/07_weekly_analysis.py"),
        Path("dev_pages/08_weekly_interpretation.py"),
        Path("dev_pages/09_weekly_feedback.py"),
    ]

    for page_path in page_paths:
        page_source = page_path.read_text(encoding="utf-8")
        assert "render_readonly_control" in page_source
        assert 'controls[2].metric("분석 주 종료일"' not in page_source


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


def test_consumption_interpretation_dev_page_renders_payment_behavior_metrics() -> None:
    """소비 해석 개발 페이지가 추출 지표 요약에 새 결제 행동 지표를 표시하는지 검증한다."""
    app = AppTest.from_file("dev_pages/05_daily_interpretation.py")

    app.run(timeout=10)

    assert len(app.exception) == 0
    assert any(subheader.value == "지출 마찰력 및 결제 밀도" for subheader in app.subheader)


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

    with patch(
        "catcher_llm.services.consumption_feedback.daily_feedback.generate_daily_feedback",
        return_value=fake_result,
    ):
        app = AppTest.from_file("dev_pages/06_daily_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "일일 피드백 생성")

    assert len(app.exception) == 0
    assert any(subheader.value == "지출 마찰력 및 결제 밀도" for subheader in app.subheader)
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)
    assert any(expander.label == "메모리/세션 컨텍스트 JSON" for expander in app.expander)


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

    with patch(
        "catcher_llm.services.consumption_feedback.weekly_feedback.generate_weekly_feedback",
        return_value=fake_result,
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

    with patch(
        "catcher_llm.services.consumption_feedback.monthly_feedback.generate_monthly_feedback",
        return_value=fake_result,
    ):
        app = AppTest.from_file("dev_pages/12_monthly_feedback.py")
        app.run(timeout=10)
        _click_button_by_label(app, "월간 피드백 생성")

    assert len(app.exception) == 0
    assert any(subheader.value == "피드백 근거" for subheader in app.subheader)
    assert any(expander.label == "사용자 프로필 JSON" for expander in app.expander)
