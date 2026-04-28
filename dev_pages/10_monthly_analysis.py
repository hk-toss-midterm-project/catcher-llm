from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.ui.date_picker import render_date_picker_styles, select_month

settings = get_settings()


def _format_amount(value: object) -> str:
    """월간 소비 분석 JSON의 금액 값을 원화 표시 문자열로 변환한다."""
    if isinstance(value, int | float):
        return f"{value:,.0f}원"
    return "-"


def _format_percent(value: object) -> str:
    """월간 소비 분석 JSON의 비율 값을 퍼센트 표시 문자열로 변환한다."""
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
    """반복 소비 누적 결과의 주요 가맹점 유형 요약을 지표 카드로 표시한다."""
    st.subheader("반복 소비 누적")
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ["delivery", "cafe", "convenience", "taxi"],
        ["배달", "카페", "편의점", "택시"],
        strict=True,
    ):
        summary = cast(JsonObject, repeat_patterns[key])
        column.metric(label, f"{summary['count']}건", _format_amount(summary["total_amount"]))


def _render_risk_summary(
    micro_spending: JsonObject,
    late_night_spending: JsonObject,
    high_spending: JsonObject,
) -> None:
    """월간 소액·야간·고액 소비 탐지 결과를 지표와 표로 표시한다."""
    st.subheader("문제 소비 누적")
    columns = st.columns(3)
    columns[0].metric(
        "소액 결제",
        _format_amount(micro_spending["total_amount"]),
        f"{micro_spending['count']}건",
    )
    columns[1].metric(
        "야간 소비",
        _format_amount(late_night_spending["total_amount"]),
        f"{late_night_spending['count']}건",
    )
    columns[2].metric(
        "고액 결제",
        _format_amount(high_spending["total_amount"]),
        f"{high_spending['count']}건",
    )

    _render_table(
        "소액 결제 상위 카테고리",
        micro_spending["top_categories"],
        "소액 결제 데이터가 없습니다.",
    )
    _render_table(
        "야간 소비 상위 카테고리",
        late_night_spending["top_categories"],
        "야간 소비 데이터가 없습니다.",
    )
    _render_table("고액 결제 항목", high_spending["items"], "고액 결제 항목이 없습니다.")


def _render_document_monthly_metrics(monthly_metrics: JsonObject) -> None:
    """문서 기준 월간 핵심 지표와 특수 지표를 화면에 표시한다."""
    st.subheader("문서 기준 월간 핵심 지표")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "월간 총 소비금액",
        _format_amount(monthly_metrics["monthly_total_amount"]),
    )
    metric_columns[1].metric(
        "월간 예산 대비 사용률",
        _format_percent(monthly_metrics["monthly_budget_usage_rate_percent"]),
    )
    metric_columns[2].metric(
        "전월 대비 소비 증감률",
        _format_percent(monthly_metrics["previous_month_change_rate_percent"]),
    )
    metric_columns[3].metric("구독료 합계", _format_amount(monthly_metrics["subscription_total"]))

    structure_columns = st.columns(4)
    structure_columns[0].metric("고정비 금액", _format_amount(monthly_metrics["fixed_cost_amount"]))
    structure_columns[1].metric(
        "고정비 비중",
        _format_percent(monthly_metrics["fixed_cost_ratio_percent"]),
    )
    structure_columns[2].metric(
        "변동비 금액", _format_amount(monthly_metrics["variable_cost_amount"])
    )
    structure_columns[3].metric(
        "고정비 부담률",
        _format_percent(monthly_metrics["fixed_cost_burden_rate_percent"]),
    )

    flow_columns = st.columns(2)
    flow_columns[0].metric(
        "급여일 이후 소비 증가율",
        _format_percent(monthly_metrics["post_salary_spending_increase_rate_percent"]),
    )
    flow_columns[1].metric(
        "월말 소비 압박 지수",
        str(monthly_metrics["month_end_pressure_index"] or "-"),
    )

    _render_table(
        "카테고리별 월간 소비 비중",
        monthly_metrics["category_monthly_spending_ratio"],
        "카테고리별 월간 소비 비중 데이터가 없습니다.",
    )


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 거래 데이터 기준 월간 소비 분석 JSON을 확인합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("📈 월간 소비 분석")
st.caption("consumption_feedback.monthly_analysis 서비스를 실행해 화면에서 결과를 점검합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_month = select_month(
        "분석 월",
        default_month="2024-04",
        key="monthly_analysis_month",
    )

if st.button("월간 분석 실행", use_container_width=True):
    with st.spinner("월간 소비 분석 JSON 생성 중..."):
        try:
            result = build_monthly_consumption_analysis_json(
                member_id=int(member_id),
                analysis_month=analysis_month,
                settings=settings,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

    monthly_summary = cast(JsonObject, result["monthly_summary"])
    monthly_metrics = cast(JsonObject, result["monthly_metrics"])
    fixed_variable = cast(JsonObject, result["fixed_variable"])
    repeat_patterns = cast(JsonObject, result["repeat_patterns"])
    weekly_trend = cast(JsonObject, result["weekly_trend"])
    micro_spending = cast(JsonObject, result["micro_spending"])
    late_night_spending = cast(JsonObject, result["late_night_spending"])
    high_spending = cast(JsonObject, result["high_spending"])
    saving_potential = cast(JsonObject, result["saving_potential"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("이번 달 총 소비", _format_amount(monthly_summary["this_month_total"]))
    metric_columns[1].metric("전월 총 소비", _format_amount(monthly_summary["prev_month_total"]))
    metric_columns[2].metric(
        "전월 대비 증감률", _format_percent(monthly_summary["diff_rate_percent"])
    )
    metric_columns[3].metric("결제 건수", f"{monthly_summary['transaction_count']}건")

    fixed_columns = st.columns(4)
    fixed_columns[0].metric("고정비", _format_amount(fixed_variable["fixed_total"]))
    fixed_columns[1].metric("변동비", _format_amount(fixed_variable["variable_total"]))
    fixed_columns[2].metric("고정비 비중", _format_percent(fixed_variable["fixed_ratio_percent"]))
    fixed_columns[3].metric("주차 추이", str(weekly_trend["trend_direction"]))

    saving_columns = st.columns(3)
    saving_columns[0].metric(
        "예상 절약 가능액",
        _format_amount(saving_potential["total_potential_saving"]),
    )
    saving_columns[1].metric(
        "다음 달 권장 목표",
        _format_amount(saving_potential["next_month_recommended_target"]),
    )
    saving_columns[2].metric("활동일", f"{monthly_summary['active_days']}일")

    _render_document_monthly_metrics(monthly_metrics)

    _render_table("카테고리 심층 분석", result["category_deep"], "카테고리 데이터가 없습니다.")
    _render_table(
        "절약 가능성 TOP 카테고리",
        result["top_savable_categories"],
        "절약 가능성 카테고리가 없습니다.",
    )
    _render_table("고정비 항목", fixed_variable["fixed_items"], "고정비 항목이 없습니다.")
    _render_repeat_summary(repeat_patterns)
    _render_table(
        "TOP 5 가맹점",
        repeat_patterns["top5_merchants"],
        "반복 소비 가맹점이 없습니다.",
    )
    _render_table(
        "주차별 소비 추이",
        weekly_trend["weekly_breakdown"],
        "주차별 소비 데이터가 없습니다.",
    )
    _render_risk_summary(micro_spending, late_night_spending, high_spending)

    st.caption(f"source_path: `{result['source_path']}` | prev_month: `{result['prev_month']}`")
    with st.expander("Raw JSON"):
        st.json(result)
