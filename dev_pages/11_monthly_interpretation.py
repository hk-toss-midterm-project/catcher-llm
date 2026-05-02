from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from catcher_llm.chains.consumption_feedback import build_monthly_spending_analysis_chain
from catcher_llm.config.settings import get_settings
from catcher_llm.schemas.consumption_feedback import (
    MonthlyCategoryChangeIndicator,
    MonthlyFixedItem,
    MonthlyHighSpendingItem,
    MonthlyMerchantVisit,
    MonthlySpendingData,
    MonthlySpendingIndicatorPayload,
    SpendingMetric,
    UserProfileContext,
)
from catcher_llm.services.consumption_feedback.daily_feedback import load_user_profile_context
from catcher_llm.services.consumption_feedback.monthly_analysis import (
    build_monthly_consumption_analysis_json,
)
from catcher_llm.services.consumption_feedback.monthly_feedback import (
    extract_monthly_spending_indicators,
    make_monthly_spending_analysis_input,
    parse_monthly_spending_data,
)
from catcher_llm.ui.date_picker import (
    DEFAULT_CALENDAR_MONTH,
    render_date_picker_styles,
    select_month,
)

settings = get_settings()


def _models_to_frame(
    models: Sequence[BaseModel],
    *,
    string_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Pydantic 모델 목록을 Streamlit 표로 렌더링할 DataFrame으로 변환한다."""
    columns_to_stringify = string_columns or set()
    rows: list[dict[str, object]] = []
    for model in models:
        row = cast(dict[str, object], model.model_dump())
        for column_name in columns_to_stringify:
            if column_name in row:
                row[column_name] = str(row[column_name])
        rows.append(row)
    return pd.DataFrame(rows)


def _format_metric_value(metric: SpendingMetric) -> str:
    """월간 소비 지표 값을 단위에 맞는 화면 표시 문자열로 변환한다."""
    value = metric.value
    if metric.unit == "KRW" and isinstance(value, int | float):
        return f"{value:,.0f}원"
    if metric.unit == "percent" and isinstance(value, int | float):
        return f"{value:,.2f}%"
    if metric.unit == "ratio" and isinstance(value, int | float):
        return f"{value:,.2f}x"
    if metric.unit == "count":
        return f"{value}건"
    return str(value)


def _find_metric(metrics: Sequence[SpendingMetric], name: str) -> SpendingMetric | None:
    """지표 이름으로 핵심 소비 지표를 찾아 화면 요약에 사용할 수 있게 반환한다."""
    for metric in metrics:
        if metric.name == name:
            return metric
    return None


def _metric_value_text(metrics: Sequence[SpendingMetric], name: str) -> str:
    """지표 이름에 해당하는 값을 Streamlit metric 값 문자열로 변환한다."""
    metric = _find_metric(metrics, name)
    if metric is None:
        return "-"
    return _format_metric_value(metric)


def _render_metric_table(metrics: Sequence[SpendingMetric]) -> None:
    """월간 핵심 소비 지표 목록을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(metrics, string_columns={"value"})
    if frame.empty:
        st.info("표시할 월간 핵심 지표가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_category_change_table(changes: Sequence[MonthlyCategoryChangeIndicator]) -> None:
    """월간 카테고리 증감 지표를 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(changes)
    if frame.empty:
        st.info("표시할 카테고리 증감 지표가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_high_spending_table(items: Sequence[MonthlyHighSpendingItem]) -> None:
    """월간 고액 결제 항목을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(items)
    if frame.empty:
        st.info("고액 결제 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_fixed_item_table(items: Sequence[MonthlyFixedItem]) -> None:
    """월간 고정비 항목을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(items)
    if frame.empty:
        st.info("고정비 항목이 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_top_merchant_table(items: Sequence[MonthlyMerchantVisit]) -> None:
    """월간 반복 가맹점 목록을 개발 확인용 표로 표시한다."""
    frame = _models_to_frame(items)
    if frame.empty:
        st.info("반복 가맹점 데이터가 없습니다.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


def _load_sqlite_monthly_data(
    *,
    member_id: int,
    analysis_month: str,
) -> MonthlySpendingData:
    """SQLite 월간 분석 JSON을 생성한 뒤 월간 해석 입력 모델로 변환한다."""
    raw_result = build_monthly_consumption_analysis_json(
        member_id=member_id,
        analysis_month=analysis_month,
        settings=settings,
    )
    return parse_monthly_spending_data(raw_result)


def _load_profile(member_id: int) -> UserProfileContext | None:
    """SQLite 사용자 프로필을 읽고 실패 시 체인 입력에서 제외할 수 있게 None을 반환한다."""
    try:
        return load_user_profile_context(member_id=member_id, settings=settings)
    except ValueError:
        return None


def _render_financial_context_summary(metrics: Sequence[SpendingMetric]) -> None:
    """소득과 목표 소비 기반 월간 해석 지표를 개발 확인용 메트릭으로 표시한다."""
    st.subheader("소득·목표 소비 지표")
    budget_columns = st.columns(4)
    budget_columns[0].metric(
        "월간 잔여 예산",
        _metric_value_text(metrics, "월간 잔여 예산"),
    )
    budget_columns[1].metric(
        "월간 예산 초과액",
        _metric_value_text(metrics, "월간 예산 초과액"),
    )
    budget_columns[2].metric(
        "월소득 대비 총소비율",
        _metric_value_text(metrics, "월소득 대비 총소비율"),
    )
    budget_columns[3].metric(
        "목표 소비 한도 소득 비중",
        _metric_value_text(metrics, "목표 소비 한도 소득 비중"),
    )

    saving_columns = st.columns(4)
    saving_columns[0].metric(
        "추정 저축액",
        _metric_value_text(metrics, "추정 저축액"),
    )
    saving_columns[1].metric(
        "추정 저축률",
        _metric_value_text(metrics, "추정 저축률"),
    )
    saving_columns[2].metric(
        "목표 달성 시 저축액",
        _metric_value_text(metrics, "목표 달성 시 저축액"),
    )
    saving_columns[3].metric(
        "목표 달성 시 저축률",
        _metric_value_text(metrics, "목표 달성 시 저축률"),
    )

    capacity_columns = st.columns(2)
    capacity_columns[0].metric(
        "소비 여력",
        _metric_value_text(metrics, "소비 여력"),
    )
    capacity_columns[1].metric(
        "비필수 소비 소득 비중",
        _metric_value_text(metrics, "비필수 소비 소득 비중"),
    )


def _render_indicator_summary(indicators: MonthlySpendingIndicatorPayload) -> None:
    """추출된 월간 소비 지표 묶음의 핵심 요약과 상세 표를 표시한다."""
    metric_columns = st.columns(5)
    metric_columns[0].metric("Member ID", indicators.member_id)
    metric_columns[1].metric("분석 월", indicators.analysis_month)
    metric_columns[2].metric(
        "최대 증가 카테고리",
        indicators.largest_category_increase.category
        if indicators.largest_category_increase
        else "-",
    )
    metric_columns[3].metric(
        "최대 감소 카테고리",
        indicators.largest_category_decrease.category
        if indicators.largest_category_decrease
        else "-",
    )
    metric_columns[4].metric("주차 추이", indicators.trend_direction)

    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "이번 달 총 지출",
        _metric_value_text(indicators.metrics, "이번 달 총 지출액"),
    )
    summary_columns[1].metric(
        "고정비 총액",
        _metric_value_text(indicators.metrics, "고정비 총액"),
    )
    summary_columns[2].metric(
        "배달 결제 금액",
        _metric_value_text(indicators.metrics, "배달 결제 금액"),
    )
    summary_columns[3].metric(
        "총 절약 가능액",
        _metric_value_text(indicators.metrics, "총 절약 가능액"),
    )

    _render_financial_context_summary(indicators.metrics)

    st.subheader("월간 핵심 소비 지표")
    _render_metric_table(indicators.metrics)

    st.subheader("월간 카테고리 증감")
    _render_category_change_table(indicators.category_changes)

    st.subheader("월간 고정비 항목")
    _render_fixed_item_table(indicators.fixed_items)

    st.subheader("월간 고액 결제 항목")
    _render_high_spending_table(indicators.high_spending_items)

    st.subheader("월간 반복 가맹점")
    _render_top_merchant_table(indicators.top_merchants)


def _render_chain_result(result: dict[str, object]) -> None:
    """월간 구조화 분석 체인 결과를 단계별 JSON으로 표시한다."""
    sections = [
        ("pattern_result", "월간 소비 패턴 분석"),
        ("problem_result", "월간 문제 소비 식별"),
        ("cause_result", "월간 소비 원인 해석"),
        ("action_result", "다음 달 행동 개선 제안"),
    ]
    for key, title in sections:
        if key not in result:
            continue
        value = result[key]
        st.subheader(title)
        if isinstance(value, BaseModel):
            st.json(value.model_dump())
        else:
            st.json(value)


with st.sidebar:
    st.title("🧪 Catcher Dev")
    st.caption("SQLite 월간 소비 분석 JSON 기반 지표 추출과 구조화 해석 체인을 점검합니다.")
    st.write(f"SQLite: `{settings.sqlite_db_path}`")


st.title("🧭 월간 소비 해석 체인")
st.caption("user_monthly_analysis 결과를 월간 지표로 변환하고 구조화 해석 체인을 실행합니다.")

render_date_picker_styles()
controls = st.columns(2)
member_id = controls[0].number_input("Member ID", min_value=1, value=1, step=1)
with controls[1]:
    analysis_month = select_month(
        "분석 월",
        default_month=DEFAULT_CALENDAR_MONTH,
        key="monthly_interpretation_month",
    )

try:
    monthly_data = _load_sqlite_monthly_data(
        member_id=int(member_id),
        analysis_month=analysis_month,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

user_profile = _load_profile(int(member_id))
indicators = extract_monthly_spending_indicators(monthly_data)
analysis_input = make_monthly_spending_analysis_input(monthly_data, user_profile=user_profile)

_render_indicator_summary(indicators)

with st.expander("체인 입력 JSON"):
    st.code(analysis_input["indicator_json"], language="json")
    st.code(analysis_input["raw_json"], language="json")
    st.code(analysis_input["user_profile_json"], language="json")

chat_model_error = settings.chat_model_error
if chat_model_error is not None:
    st.warning(
        settings.get_chat_model_error_message("monthly consumption interpretation flow")
        or "Chat model configuration is invalid."
    )
elif st.button("월간 구조화 분석 체인 실행", width="stretch"):
    with st.spinner("월간 소비 해석 체인 실행 중..."):
        try:
            chain = build_monthly_spending_analysis_chain(settings=settings)
            chain_result = chain.invoke(analysis_input)
        except Exception as error:
            st.error(f"체인 실행에 실패했습니다: {error}")
            st.stop()

    _render_chain_result(chain_result)
