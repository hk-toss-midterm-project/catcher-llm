from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.weekly_analysis import (
    build_weekly_consumption_analysis_json,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_DATE,
    render_date_picker_styles,
    select_week_range,
)

settings = get_settings()


def _format_amount(value: object) -> str:
    """주간 소비 분석 JSON의 금액 값을 원화 표시 문자열로 변환한다."""
    if isinstance(value, int | float):
        return f"{value:,.0f}원"
    return "-"


def _format_percent(value: object) -> str:
    """주간 소비 분석 JSON의 비율 값을 퍼센트 표시 문자열로 변환한다."""
    if isinstance(value, int | float):
        return f"{value:,.2f}%"
    return "-"


def _json_rows_to_frame(rows: object) -> pd.DataFrame:
    """JSON 목록 값을 Streamlit 표 렌더링에 사용할 DataFrame으로 변환한다."""
    if not isinstance(rows, list):
        return pd.DataFrame()
    json_rows = [row for row in rows if isinstance(row, dict)]
    return pd.DataFrame(json_rows)


def _render_table(title: str, rows: object, empty_message: str) -> None:
    """제목과 빈 상태 메시지를 포함해 JSON 목록을 표로 표시한다."""
    st.subheader(title)
    frame = _json_rows_to_frame(rows)
    if frame.empty:
        st.info(empty_message)
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_repeat_summary(repeat_patterns: JsonObject) -> None:
    """반복 소비 패턴의 주요 가맹점 유형 요약을 지표 카드로 표시한다."""
    st.subheader("반복 소비 요약")
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ["delivery", "cafe", "convenience", "taxi"],
        ["배달", "카페", "편의점", "택시"],
        strict=True,
    ):
        summary = cast(JsonObject, repeat_patterns[key])
        column.metric(label, f"{summary['count']}건", _format_amount(summary["total_amount"]))


def _render_waste_summary(waste_detection: JsonObject) -> None:
    """낭비성 소비 탐지 결과의 핵심 지표와 상세 항목을 표시한다."""
    late_night = cast(JsonObject, waste_detection["late_night"])
    micro_spending = cast(JsonObject, waste_detection["micro_spending"])
    high_spending = cast(JsonObject, waste_detection["high_spending"])

    st.subheader("낭비성 소비 탐지")
    columns = st.columns(3)
    columns[0].metric(
        "야간 소비",
        _format_amount(late_night["total_amount"]),
        f"{late_night['count']}건",
    )
    columns[1].metric(
        "소액 결제",
        _format_amount(micro_spending["total_amount"]),
        f"{micro_spending['count']}건",
    )
    columns[2].metric(
        "고액 결제",
        _format_amount(high_spending["total_amount"]),
        f"{high_spending['count']}건",
    )

    _render_table("고액 결제 항목", high_spending["items"], "고액 결제 항목이 없습니다.")
    _render_table("야간 소비 상위 카테고리", late_night["top_categories"], "야간 소비가 없습니다.")
    _render_table(
        "소액 결제 상위 카테고리",
        micro_spending["top_categories"],
        "소액 결제가 없습니다.",
    )


def _render_document_weekly_metrics(weekly_metrics: JsonObject) -> None:
    """기존 요약과 겹치지 않는 문서 기준 주간 추가 지표를 화면에 표시한다."""
    st.subheader("문서 기준 주간 추가 지표")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "주중 소비 비중",
        _format_percent(weekly_metrics["weekday_spending_ratio_percent"]),
    )
    metric_columns[1].metric(
        "주말 소비 비중",
        _format_percent(weekly_metrics["weekend_spending_ratio_percent"]),
    )
    metric_columns[2].metric(
        "주간 소비 변동성",
        _format_amount(weekly_metrics["weekly_spending_volatility"]),
    )
    special_metrics = cast(JsonObject, weekly_metrics["special_metrics"])
    special_columns = st.columns(3)
    special_columns[0].metric(
        "주간 예산 소진율",
        _format_percent(weekly_metrics["weekly_budget_usage_rate_percent"]),
    )
    special_columns[1].metric(
        "주말 과소비 지수",
        str(special_metrics["weekend_overspending_index"]),
    )
    special_columns[2].metric(
        "소비 요일 편중도",
        _format_percent(special_metrics["weekday_concentration_ratio_percent"]),
    )


def _render_weekly_comparison_card(
    column: DeltaGenerator,
    label: str,
    comparison: JsonObject,
) -> None:
    """주간 단일 기준 기간 비교 결과를 증감액과 기준 기간이 있는 카드로 표시한다."""
    column.metric(
        label,
        _format_amount(comparison["amount_diff"]),
        _format_percent(comparison["amount_diff_rate_percent"]),
    )
    column.caption(
        f"{comparison['reference_start_date']}~{comparison['reference_end_date']} · "
        f"{_format_amount(comparison['reference_total'])} · "
        f"{comparison['reference_count']}건"
    )


def _render_weekly_comparisons(weekly_comparisons: JsonObject) -> None:
    """전주·최근 4주 평균·지난달 같은 주차 비교를 화면에 표시한다."""
    st.subheader("확장 주간 비교")
    previous_week = cast(JsonObject, weekly_comparisons["previous_week"])
    recent_average = cast(JsonObject, weekly_comparisons["recent_4week_average"])
    same_week_last_month = cast(JsonObject, weekly_comparisons["same_week_last_month"])

    columns = st.columns(3)
    _render_weekly_comparison_card(columns[0], "전주 대비", previous_week)
    columns[1].metric(
        "최근 4주 평균 대비",
        _format_amount(recent_average["amount_diff"]),
        _format_percent(recent_average["amount_diff_rate_percent"]),
    )
    columns[1].caption(
        f"기준 {recent_average['reference_week_count']}주 평균 · "
        f"{_format_amount(recent_average['average_total'])} · "
        f"{recent_average['average_count']}건"
    )
    _render_weekly_comparison_card(columns[2], "지난달 같은 주차 대비", same_week_last_month)

    reference_period_frame = _json_rows_to_frame(recent_average["reference_periods"])
    if reference_period_frame.empty:
        st.info("최근 4주 평균에 사용할 기준 주간 데이터가 없습니다.")
    else:
        st.dataframe(reference_period_frame, use_container_width=True, hide_index=True)


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 거래 데이터 기준 주간 소비 분석 JSON을 확인합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("🗓️ 주간 소비 분석")
st.caption("consumption_feedback.weekly_analysis 서비스를 실행해 화면에서 결과를 점검합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    week_start, week_end = select_week_range(
        "분석 주",
        default_start=DEFAULT_CALENDAR_DATE,
        key="weekly_analysis_week",
    )

if st.button("주간 분석 실행", use_container_width=True):
    with st.spinner("주간 소비 분석 JSON 생성 중..."):
        try:
            result = build_weekly_consumption_analysis_json(
                member_id=int(member_id),
                week_start=week_start,
                week_end=week_end,
                settings=settings,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    weekly_summary = cast(JsonObject, result["weekly_summary"])
    weekly_comparisons = cast(JsonObject, result["weekly_comparisons"])
    weekly_metrics = cast(JsonObject, result["weekly_metrics"])
    repeat_patterns = cast(JsonObject, result["repeat_patterns"])
    weekday_pattern = cast(JsonObject, result["weekday_pattern"])
    waste_detection = cast(JsonObject, result["waste_detection"])
    saving_potential = cast(JsonObject, result["saving_potential"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("이번 주 총 소비", _format_amount(weekly_summary["this_week_total"]))
    metric_columns[1].metric("전주 총 소비", _format_amount(weekly_summary["prev_week_total"]))
    metric_columns[2].metric(
        "전주 대비 증감률", _format_percent(weekly_summary["diff_rate_percent"])
    )
    metric_columns[3].metric("결제 건수", f"{weekly_summary['transaction_count']}건")

    peak_columns = st.columns(3)
    peak_columns[0].metric("일평균", _format_amount(weekly_summary["daily_average"]))
    peak_columns[1].metric("최대 소비일", str(weekly_summary["max_day_date"] or "-"))
    peak_columns[2].metric("피크 요일", str(weekday_pattern["peak_weekday"] or "-"))

    _render_document_weekly_metrics(weekly_metrics)
    _render_weekly_comparisons(weekly_comparisons)

    _render_table("카테고리별 주간 분석", result["category_summary"], "카테고리 데이터가 없습니다.")
    _render_repeat_summary(repeat_patterns)
    _render_table(
        "TOP 가맹점",
        repeat_patterns["top_merchants"],
        "반복 소비 가맹점이 없습니다.",
    )
    _render_table(
        "연속 소비 가맹점",
        repeat_patterns["consecutive_merchants"],
        "연속 소비 가맹점이 없습니다.",
    )
    _render_table(
        "요일별 소비 패턴",
        weekday_pattern["weekday_breakdown"],
        "요일별 소비 데이터가 없습니다.",
    )
    _render_waste_summary(waste_detection)
    _render_table(
        "개선된 카테고리",
        saving_potential["improved_categories"],
        "전주 대비 개선된 카테고리가 없습니다.",
    )
    _render_table(
        "악화된 카테고리",
        saving_potential["worsened_categories"],
        "전주 대비 악화된 카테고리가 없습니다.",
    )

    st.caption(f"source_path: `{result['source_path']}`")
    with st.expander("Raw JSON"):
        st.json(result)
