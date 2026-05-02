from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import JsonObject
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)

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
    st.dataframe(frame, width="stretch", hide_index=True)


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
    """기존 요약과 겹치지 않는 문서 기준 월간 추가 지표를 화면에 표시한다."""
    st.subheader("문서 기준 월간 추가 지표")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "월간 예산 대비 사용률",
        _format_percent(monthly_metrics["monthly_budget_usage_rate_percent"]),
    )
    metric_columns[1].metric(
        "월간 잔여 예산",
        _format_amount(monthly_metrics.get("monthly_remaining_budget")),
    )
    metric_columns[2].metric(
        "월간 예산 초과액",
        _format_amount(monthly_metrics.get("monthly_overspend_amount")),
    )
    metric_columns[3].metric("구독료 합계", _format_amount(monthly_metrics["subscription_total"]))

    income_columns = st.columns(4)
    income_columns[0].metric(
        "월소득 대비 총소비율",
        _format_percent(monthly_metrics.get("monthly_income_usage_rate_percent")),
    )
    income_columns[1].metric(
        "목표 소비 한도 소득 비중",
        _format_percent(monthly_metrics.get("target_spending_to_income_rate_percent")),
    )
    income_columns[2].metric(
        "고정비 부담률",
        _format_percent(monthly_metrics["fixed_cost_burden_rate_percent"]),
    )
    income_columns[3].metric(
        "비필수 소비 소득 비중",
        _format_percent(monthly_metrics.get("nonessential_spending_income_rate_percent")),
    )

    saving_columns = st.columns(4)
    saving_columns[0].metric(
        "추정 저축액",
        _format_amount(monthly_metrics.get("estimated_saving_amount")),
    )
    saving_columns[1].metric(
        "추정 저축률",
        _format_percent(monthly_metrics.get("estimated_saving_rate_percent")),
    )
    saving_columns[2].metric(
        "목표 달성 시 저축액",
        _format_amount(monthly_metrics.get("target_saving_amount")),
    )
    saving_columns[3].metric(
        "목표 달성 시 저축률",
        _format_percent(monthly_metrics.get("target_saving_rate_percent")),
    )

    pressure_columns = st.columns(3)
    pressure_columns[0].metric(
        "소비 여력",
        _format_amount(monthly_metrics.get("spending_capacity")),
    )
    pressure_columns[1].metric(
        "급여일 이후 소비 증가율",
        _format_percent(monthly_metrics["post_salary_spending_increase_rate_percent"]),
    )
    pressure_columns[2].metric(
        "월말 소비 압박 지수",
        str(monthly_metrics["month_end_pressure_index"] or "-"),
    )


def _render_monthly_comparison_card(
    column: DeltaGenerator,
    label: str,
    comparison: JsonObject,
) -> None:
    """월간 단일 기준월 비교 결과를 증감액과 기준월 맥락이 있는 카드로 표시한다."""
    column.metric(
        label,
        _format_amount(comparison["amount_diff"]),
        _format_percent(comparison["amount_diff_rate_percent"]),
    )
    column.caption(
        f"기준 {comparison['reference_month']} · "
        f"{_format_amount(comparison['reference_total'])} · "
        f"{comparison['reference_count']}건"
    )


def _render_monthly_comparisons(monthly_comparisons: JsonObject) -> None:
    """전월·최근 3개월 평균 비교를 화면에 표시한다."""
    st.subheader("확장 월간 비교")
    previous_month = cast(JsonObject, monthly_comparisons["previous_month"])
    recent_average = cast(JsonObject, monthly_comparisons["recent_3month_average"])

    columns = st.columns(2)
    _render_monthly_comparison_card(columns[0], "전월 대비", previous_month)
    columns[1].metric(
        "최근 3개월 평균 대비",
        _format_amount(recent_average["amount_diff"]),
        _format_percent(recent_average["amount_diff_rate_percent"]),
    )
    columns[1].caption(
        f"기준 {recent_average['reference_month_count']}개월 평균 · "
        f"{_format_amount(recent_average['average_total'])} · "
        f"{recent_average['average_count']}건"
    )

    reference_month_frame = _json_rows_to_frame(recent_average["reference_month_details"])
    if reference_month_frame.empty:
        st.info("최근 3개월 평균에 사용할 기준월 데이터가 없습니다.")
    else:
        st.dataframe(reference_month_frame, width="stretch", hide_index=True)


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
        default_month=DEFAULT_CALENDAR_MONTH,
        key="monthly_analysis_month",
    )

if st.button("월간 분석 실행", width="stretch"):
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
    monthly_comparisons = cast(JsonObject, result["monthly_comparisons"])
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
    _render_monthly_comparisons(monthly_comparisons)

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
